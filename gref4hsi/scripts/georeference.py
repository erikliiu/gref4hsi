# Built-ins
import configparser
import json
import os
import sys

# Third party
from scipy.spatial.transform import Rotation as RotLib
import numpy as np
import pyvista as pv
import h5py

# Lib resources:
from gref4hsi.utils.geometry_utils import CameraGeometry, CalibHSI
from gref4hsi.utils.parsing_utils import Hyperspectral
from gref4hsi.utils import visualize


def cal_file_to_rays(filename_cal):
    # See paper by Sun, Bo, et al. "Calibration of line-scan cameras for precision measurement." Applied optics 55.25 (2016): 6836-6843.
    # Loads line camera parameters for the hyperspectral imager from an xml file.

    # Certain imagers deliver geometry "per pixel". This can be resolved by fitting model parameters.
    calHSI = CalibHSI(file_name_cal_xml=filename_cal)
    f = calHSI.f
    u_c = calHSI.cx

    # Radial distortion parameters
    k1 = calHSI.k1
    k2 = calHSI.k2

    # Tangential distortion parameters
    k3 = calHSI.k3

    # Translation (lever arm) of HSI with respect to vehicle frame
    trans_x = calHSI.tx
    trans_y = calHSI.ty
    trans_z = calHSI.tz

    # Rotation of HSI (boresight) with respect to vehicle navigation frame (often defined by IMU or RGB cam)
    rot_x = calHSI.rx
    rot_y = calHSI.ry
    rot_z = calHSI.rz

    # Number of pixels
    n_pix = calHSI.w

    # Define camera model array.
    u = np.arange(1, n_pix + 1)

    # Express uhi ray directions in uhi frame using line-camera model
    x_norm_lin = (u - u_c) / f

    x_norm_nonlin = (
        -(
            k1 * ((u - u_c) / 1000) ** 5
            + k2 * ((u - u_c) / 1000) ** 3
            + k3 * ((u - u_c) / 1000) ** 2
        )
        / f
    )

    x_norm = x_norm_lin + x_norm_nonlin

    p_dir = np.zeros((len(x_norm), 3))

    # Rays are defined in the HI frame with positive z down
    p_dir[:, 0] = x_norm
    p_dir[:, 2] = 1

    rot_hsi_ref_eul = np.array([rot_z, rot_y, rot_x])

    rot_hsi_ref_obj = RotLib.from_euler(
        seq="ZYX", angles=rot_hsi_ref_eul, degrees=False
    )

    translation_ref_hsi = np.array([trans_x, trans_y, trans_z])

    intrinsic_geometry_dict = {
        "translation_ref_hsi": translation_ref_hsi,
        "rot_hsi_ref_obj": rot_hsi_ref_obj,
        "ray_directions_local": p_dir,
    }

    # Notably, one could compress the information by expressing the ray directions in the body frame

    return intrinsic_geometry_dict


def define_hsi_ray_geometry(
    pos_ref_ecef,
    quat_ref_ecef,
    time_pose,
    intrinsic_geometry_dict,
):
    """
    # Input:
    # - pos_ref_ecef: camera positions (ECEF coordinates)
    # - quat_ref_ecef: camera orientations (as quaternions)
    # - time_pose: timestamps for the poses
    # - intrinsic_geometry_dict:
    #   * lever arm (translation)
    #   * boresight rotation (rotation)
    #   * ray directions in local camera frame
    # What it does:
    # - Create a CameraGeometry object
    # - Corrects camera position and orientation with lever arm + boresight
    # - Defines how each pixel ray points out globally
    # Output:
    # - CameraGeometry object
    #   (has .position_nav, .rotation_nav, .rayDirectionsGlobal, etc.)"""

    """Instantiate a camera geometry object from the h5 pose data"""

    pos = pos_ref_ecef  # Reference positions in ECEF
    rot_obj = RotLib.from_quat(quat_ref_ecef)  # Reference orientations wrt ECEF

    ray_directions_local = intrinsic_geometry_dict["ray_directions_local"]
    translation_ref_hsi = intrinsic_geometry_dict["translation_ref_hsi"]
    rot_hsi_ref_obj = intrinsic_geometry_dict["rot_hsi_ref_obj"]

    hsi_geometry = CameraGeometry(
        pos=pos, rot=rot_obj, time=time_pose, is_interpolated=True
    )

    hsi_geometry.intrinsicTransformHSI(
        translation_ref_hsi=translation_ref_hsi, rot_hsi_ref_obj=rot_hsi_ref_obj
    )

    hsi_geometry.defineRayDirections(dir_local=ray_directions_local)

    return hsi_geometry


def write_intersection_geometry_2_h5_file(hsi_geometry, config, h5_filename):
    # Write all intersection data (ancilliary) that could be relevant

    dict_ancilliary = config["Georeferencing"]
    # Dictionary keys correspond to CameraGeometry attribute names (e.g. hsi_geometry.key), while values correspond to h5 data set paths.

    with h5py.File(h5_filename, "a", libver="latest") as f:
        for attribute_name, h5_hierarchy_item_path in dict_ancilliary.items():
            if attribute_name != "folder":
                if h5_hierarchy_item_path in f:
                    del f[h5_hierarchy_item_path]
                dset = f.create_dataset(
                    name=h5_hierarchy_item_path,
                    data=getattr(hsi_geometry, attribute_name),
                )


# Function called to apply standard processing on a folder of files
def main(iniPath, viz=False, use_coreg_param=False):
    """
    It does not return anything.
    It saves everything directly into the H5 file you loaded.

    Step-by-step:

    1. Reads the hyperspectral H5 file.

    2. Adds new datasets inside that same H5 file:
    - Intersection points
    - Normals
    - View angles
    - Sun angles
    - Tide corrections
    - Seabed depth

    3. Updates the H5 file on disk.

    """
    config = configparser.ConfigParser()
    config.read(iniPath)

    # Paths to 3D mesh ply file
    # created by the export model script earlier that took the altitude dem and turnined it into a mesh (.ply)
    path_mesh = config["Absolute Paths"]["model_path"]

    # Directory of H5 files
    dir_r = config["Absolute Paths"]["h5_folder"]

    # Timestamps here
    h5_folder_time_pose = config["HDF.processed_nav"]["timestamp"]

    # Use the regular parameters from nav system and manufacturer:
    # The path to the XML file
    """
    This XML is a **calibration file** for a line-scan hyperspectral camera.
    It describes the camera’s internal geometry and lens distortions.
    Each tag gives one important number:

    <rx>, <ry>, <rz>: 
        - Rotation of the camera relative to the vehicle body (radians).
        - rx = rotation about x-axis, ry = about y-axis, rz = about z-axis.
        - Here rz = -1.5708 rad ≈ -90°, meaning the camera is rotated 90° around the z-axis.

    <tx>, <ty>, <tz>:
        - Translation of the camera relative to the vehicle (in meters).
        - tx = ty = tz = 0, meaning no translation (camera at body frame origin).

    <f>:
        - Focal length of the camera in pixels.
        - Here, f = 930.817 pixels (important for calculating field of view).

    <cx>:
        - Optical center (principal point) of the camera in the x-direction (pixels).
        - cx = 484.25 means pixel 484 is where the center ray is expected.

    <k1>, <k2>, <k3>:
        - Lens distortion coefficients (for barrel/pincushion distortion).
        - Very small values here → almost no distortion.

    <width>:
        - Number of pixels in the camera line sensor (horizontal).
        - 968 pixels across the sensor width.

    Summary:
    - This XML defines the **rotation**, **position**, **lens properties**, and **pixel size** of the hyperspectral camera.
    - It is used to **project each pixel into the real world** during georeferencing.
    """

    hsi_cal_xml = config["Absolute Paths"]["hsi_calib_path"]

    # Position is stored here in the H5 file
    h5_folder_position_ecef = config["HDF.processed_nav"]["position_ecef"]

    # Quaternion is stored here in the H5 file
    h5_folder_quaternion_ecef = config["HDF.processed_nav"]["quaternion_ecef"]

    # not relevant for me, nromally set to false
    # if use_coreg_param:
    #     print("Using coregistred parameters for georeferencing")

    #     # Set the camera model to the calibrated one if it exists
    #     hsi_cal_xml_coreg = config["Absolute Paths"]["calib_file_coreg"]
    #     if os.path.exists(hsi_cal_xml_coreg):
    #         hsi_cal_xml = hsi_cal_xml_coreg

    #     # Optimized position
    #     h5_folder_position_ecef_coreg = config["HDF.coregistration"]["position_ecef"]

    #     # Quaternion
    #     h5_folder_quaternion_ecef_coreg = config["HDF.coregistration"][
    #         "quaternion_ecef"
    #     ]

    # The path to the Tide file (if necessary and available)
    # Leo dont have this
    try:
        path_tide = config["Absolute Paths"]["tide_path"]
    except Exception as e:
        path_tide = "Undefined"

    # Maximal allowed ray length -> just so it dont shoot to infinity
    max_ray_length = float(config["General"]["max_ray_length"])

    mesh = pv.read(path_mesh)

    """metadata_mesh = {
        "offset_x": offset_x,
        "offset_y": offset_y,
        "offset_z": offset_y,
        "epsg_code": geocsc.to_epsg(),  # Example EPSG code, replace with your actual code
        "data_type": str(mesh.points.dtype),  # Add other metadata entries here
    }"""

    """
    The model_meta.json stores how far the model's center is from the Earth's center (ECEF origin).
    These offsets (offset_x, offset_y, offset_z) are loaded and stored as mesh_trans.
    They are used to properly re-position the hyperspectral points back into global coordinates during georeferencing.
    """

    model_meta_path = path_mesh.split(".")[0] + "_meta.json"
    with open(model_meta_path, "r") as f:
        # Load the JSON data from the file
        metadata_mesh = json.load(f)
        mesh_off_x = metadata_mesh["offset_x"]
        mesh_off_y = metadata_mesh["offset_y"]
        mesh_off_z = metadata_mesh["offset_z"]
        # Mesh is translated by this much
        mesh_trans = np.array([mesh_off_x, mesh_off_y, mesh_off_z]).astype(np.float64)

    print("\n################ Georeferencing: ################")
    # lists all .h5 files in directory named dir_r?
    files = sorted(os.listdir(dir_r))
    # Filter out files that do not end with ".h5"
    h5_files = [file for file in files if file.endswith(".h5")]
    n_files = len(h5_files)
    file_count = 0
    for filename in h5_files:
        if filename.endswith("h5") or filename.endswith("hdf"):

            progress_perc = 100 * file_count / n_files
            print(
                f"\n Georeferencing file {file_count+1}/{n_files}, progress is {progress_perc} % \n"
            )

            # Path to hierarchical file
            h5_filename = dir_r + filename

            # Read h5 file
            hyp = Hyperspectral(h5_filename, config)
            # ¨normally set to false so just skiip this condition
            if use_coreg_param:
                try:
                    # Use the coregistred dataset if it exists
                    print("Using coregistred position")
                    pos_ref_ecef = Hyperspectral.get_dataset(
                        h5_filename=h5_filename,
                        dataset_name=h5_folder_position_ecef_coreg,
                    )
                except:
                    # If not use the original
                    pos_ref_ecef = Hyperspectral.get_dataset(
                        h5_filename=h5_filename, dataset_name=h5_folder_position_ecef
                    )
                try:
                    # Use the coregistred quaternion-dataset if it exists
                    print("Using coregistred quaternion")
                    quat_ref_ecef = Hyperspectral.get_dataset(
                        h5_filename=h5_filename,
                        dataset_name=h5_folder_quaternion_ecef_coreg,
                    )
                except:
                    # If not use the original
                    quat_ref_ecef = Hyperspectral.get_dataset(
                        h5_filename=h5_filename, dataset_name=h5_folder_quaternion_ecef
                    )
            else:  # we do this else!!! Liu here

                # Just use the regular navigation data
                # If not use the original
                """
                this is the camera pose being loaded, its not IMU data
                """
                pos_ref_ecef = Hyperspectral.get_dataset(
                    h5_filename=h5_filename, dataset_name=h5_folder_position_ecef
                )
                # Extract the ecef orientations for each frame
                quat_ref_ecef = Hyperspectral.get_dataset(
                    h5_filename=h5_filename, dataset_name=h5_folder_quaternion_ecef
                )
            # Extract the timestamps for each frame
            time_pose = Hyperspectral.get_dataset(
                h5_filename=h5_filename, dataset_name=h5_folder_time_pose
            )

            # boresight is the angle between the IMu frame and the camera frame (guess it is calcualted earlier from euler angles and the lever arms)
            # Using the cal file, we can define lever arm, boresight and camera model geometry (in dictionary)
            """
            Summary of cal_file_to_rays(filename_cal):

            Inputs:
            - filename_cal: Path to the hyperspectral imager's calibration XML file.

            What it does:
            1. Reads the calibration parameters:
            - Focal length (f)
            - Principal point (u_c)
            - Distortion parameters (k1, k2, k3)
            - Translation (tx, ty, tz) from vehicle frame to HSI (lever arm)
            - Rotation angles (rx, ry, rz) from vehicle frame to HSI (boresight)
            - Image width (number of pixels)

            2. Calculates ray directions:
            - For each pixel (u = 1 to width), it computes a normalized x-coordinate:
                - First linear model (simple pinhole camera assumption)
                - Then adds nonlinear distortion corrections
            - Forms local ray directions: (x_norm, 0, 1) for each pixel.

            3. Packages into dictionary:
            - 'translation_ref_hsi': lever arm (numpy array [tx, ty, tz])
            - 'rot_hsi_ref_obj': boresight rotation as a Rotation object
            - 'ray_directions_local': local ray directions for all pixels (numpy array)

            Output:
            - Returns a dictionary ('intrinsic_geometry_dict') containing the translation, rotation, and calculated ray directions.
            """

            intrinsic_geometry_dict = cal_file_to_rays(filename_cal=hsi_cal_xml)

            # Define the rays in ECEF for each frame.
            # Input: positions, orientations, times, and camera calibration
            # Does: corrects pose (lever arm + boresight) and defines pixel ray directions
            # Output: CameraGeometry object with global ray directions

            """
            transforms rays from local camera frame to global ECEF frame.
            """
            hsi_geometry = define_hsi_ray_geometry(
                pos_ref_ecef,
                quat_ref_ecef,
                time_pose,
                intrinsic_geometry_dict=intrinsic_geometry_dict,
            )

            """
            for each ray from the camera, it finds the intersection with the mesh (.ply file) calcualted earlier. 
            results in a list of intersection points (in ECEF coordinates) and the corresponding ray directions.
            """
            hsi_geometry.intersect_with_mesh(
                mesh=mesh, max_ray_length=max_ray_length, mesh_trans=mesh_trans
            )

            # Computes the view angles in the local NED. Computationally intensive as local NED is defined for each intersection
            """
            transforms the ray directions from ECEF to local NED frame.
            """
            hsi_geometry.compute_view_directions_local_tangent_plane()

            """
            liu commenbt: i am pretty tired, and i tried to understand up to this point, rest should by trivial 
            """
            # Computes the sun angles in the local NED. Computationally intensive as local NED is defined for each intersection
            hsi_geometry.compute_sun_angles_local_tangent_plane()

            hsi_geometry.compute_tide_level(path_tide, tide_format="NMA")

            hsi_geometry.compute_elevation_mean_sealevel(
                source_epsg=config["Coordinate Reference Systems"][
                    "geocsc_epsg_export"
                ],
                geoid_path=config["Absolute Paths"]["geoid_path"],
            )

            write_intersection_geometry_2_h5_file(
                hsi_geometry=hsi_geometry, config=config, h5_filename=h5_filename
            )

            hsi_geometry.write_rgb_point_cloud(
                config=config,
                hyp=hyp,
                transect_string=filename.split(".")[0],
                mesh_trans=mesh_trans,
            )

            if viz:
                visualize.show_projected_hsi_points(
                    HSICameraGeometry=hsi_geometry,
                    config=config,
                    transect_string=filename.split(".")[0],
                    mesh_trans=mesh_trans,
                )

            # from scripts import visualize
            # visualize.show_projected_hsi_points(HSICameraGeometry=hsi_geometry, config=config, transect_string = filename.split('.')[0])

        file_count += 1


if __name__ == "__main__":
    args = sys.argv[1:]
    iniPath = args[0]
    main(iniPath)
