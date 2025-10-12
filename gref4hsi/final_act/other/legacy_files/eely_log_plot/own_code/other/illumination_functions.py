"""
functions used in the illumination correction notebook
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr


def describe_data(data, track_times=None, start_time=None, end_time=None):
    """
    Prints out general information about the 3D data array.
    """
    n_tracks, n_slit_pixels, n_wavelengths = data.shape
    print("=== Data Overview ===")
    print(f"Shape of data: {data.shape} (tracks, slit_pixels, wavelengths)")
    print(f" - Number of tracks (time steps): {n_tracks}")
    print(f" - Number of slit pixels: {n_slit_pixels}")
    print(f" - Number of wavelengths: {n_wavelengths}")
    data_min = np.min(data)
    data_max = np.max(data)
    print(f"Global Intensity Range: {data_min:.3f} to {data_max:.3f}")
    if track_times is not None:
        if len(track_times) != n_tracks:
            print("\n[Warning] The length of 'track_times' does not match 'n_tracks'!")
        else:
            print(
                f"\nTrack times (first few): {track_times[:3]} ... {track_times[-3:]}"
            )
            print(f"Start track time: {track_times[0]}")
            print(f"End track time:   {track_times[-1]}")
    if start_time is not None:
        print(f"\nOverall start time: {start_time}")
    if end_time is not None:
        print(f"Overall end time:   {end_time}")


def plot_mean_intensity(data, sigma=5, track_index=0):
    """
    Plots the intensity profile along the slit for a single track, with smoothing.
    """
    single_slit = data[track_index, :, :]
    profile = np.mean(single_slit, axis=1)
    smoothed_profile = gaussian_filter1d(profile, sigma=sigma)
    max_index = np.argmax(smoothed_profile)
    max_value = smoothed_profile[max_index]
    plt.figure(figsize=(8, 4))
    plt.plot(profile, label="Original Profile", alpha=0.5)
    plt.plot(smoothed_profile, label=f"Smoothed Profile (sigma={sigma})", linewidth=2)
    plt.scatter(max_index, max_value, color="red", zorder=3, label="Max Intensity")
    plt.text(
        max_index,
        max_value,
        f"{max_value:.2f}",
        color="red",
        fontsize=12,
        verticalalignment="bottom",
    )
    plt.xlabel("Slit Pixel (Spatial Position)")
    plt.ylabel("Average Intensity")
    plt.legend()
    plt.title("Illumination Profile Along the Slit")
    plt.show()


def plot_wavelength_vs_mean(
    data, wavelengths, desired_wavelength, sigma=5, track_index=0
):
    """
    Plots the intensity profile for a specific wavelength (in nm) alongside the broadband mean.
    Interpolates the intensity if the desired wavelength is not exactly in the wavelengths array.
    """
    single_slit = data[track_index, :, :]  # (n_slit_pixels, n_wavelengths)
    n_slit_pixels = single_slit.shape[0]
    # For each slit pixel, interpolate to get the intensity at the desired wavelength.
    intensity_profile = np.array(
        [
            np.interp(desired_wavelength, wavelengths, single_slit[i, :])
            for i in range(n_slit_pixels)
        ]
    )
    mean_profile = np.mean(single_slit, axis=1)
    smoothed_intensity = gaussian_filter1d(intensity_profile, sigma=sigma)
    smoothed_mean = gaussian_filter1d(mean_profile, sigma=sigma)

    plt.figure(figsize=(8, 4))
    plt.plot(mean_profile, label="Original Mean Profile", linestyle="--", alpha=0.7)
    plt.plot(smoothed_mean, label=f"Smoothed Mean Profile (sigma={sigma})", linewidth=2)
    plt.plot(
        intensity_profile,
        label=f"Original Intensity at {desired_wavelength} nm",
        linestyle="--",
        alpha=0.7,
    )
    plt.plot(
        smoothed_intensity,
        label=f"Smoothed Intensity at {desired_wavelength} nm (sigma={sigma})",
        linewidth=2,
    )
    plt.xlabel("Slit Pixel (Spatial Position)")
    plt.ylabel("Intensity")
    plt.title(
        f"Intensity Profiles at {desired_wavelength} nm vs. Broadband Mean (Track {track_index})"
    )
    plt.legend()
    plt.show()


def plot_wavelength_intensity(
    data, wavelengths, desired_wavelength, sigma=5, track_index=0
):
    """
    Plots the intensity profile for a specific wavelength (in nm) along the slit pixels.
    Uses interpolation if the desired wavelength is not exactly in the wavelengths array.

    Parameters:
    - data: 3D numpy array with shape (n_tracks, n_slit_pixels, n_wavelengths)
    - wavelengths: 1D numpy array of actual wavelength values (length = n_wavelengths)
    - desired_wavelength: float, the wavelength (in nm) to extract intensity for
    - sigma: float, standard deviation for Gaussian smoothing (default is 5)
    - track_index: int, index of the track (time step) to analyze (default is 0)
    """
    single_slit = data[track_index, :, :]  # (n_slit_pixels, n_wavelengths)
    n_slit_pixels = single_slit.shape[0]

    # Interpolate intensity at the desired wavelength for each slit pixel
    intensity_profile = np.array(
        [
            np.interp(desired_wavelength, wavelengths, single_slit[i, :])
            for i in range(n_slit_pixels)
        ]
    )

    # Apply Gaussian smoothing
    smoothed_intensity = gaussian_filter1d(intensity_profile, sigma=sigma)

    # Plotting
    plt.figure(figsize=(8, 4))
    plt.plot(
        intensity_profile,
        label=f"Original Intensity at {desired_wavelength} nm",
        linestyle="--",
        alpha=0.7,
    )
    plt.plot(
        smoothed_intensity,
        label=f"Smoothed Intensity at {desired_wavelength} nm (sigma={sigma})",
        linewidth=2,
    )
    plt.xlabel("Slit Pixel (Spatial Position)")
    plt.ylabel("Intensity")
    plt.title(f"Intensity Profile at {desired_wavelength} nm (Track {track_index})")
    plt.legend()
    plt.show()


def compare_wavelengths_to_mean(data, wavelengths, sigma=5, track_index=0):
    """
    Computes and plots the Pearson correlation coefficient between each wavelength's profile
    and the broadband mean profile. The x-axis uses actual wavelength values.
    """
    single_slit = data[track_index, :, :]
    mean_profile = np.mean(single_slit, axis=1)
    smoothed_mean = gaussian_filter1d(mean_profile, sigma=sigma)
    n_wavelengths = single_slit.shape[1]
    correlations = []
    for i in range(n_wavelengths):
        wavelength_profile = single_slit[:, i]
        smoothed_wavelength = gaussian_filter1d(wavelength_profile, sigma=sigma)
        corr, _ = pearsonr(smoothed_wavelength, smoothed_mean)
        correlations.append(corr)
    correlations = np.array(correlations)
    plt.figure(figsize=(10, 4))
    plt.bar(
        wavelengths,
        correlations,
        width=(wavelengths[1] - wavelengths[0]) if len(wavelengths) > 1 else 1.0,
    )
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Pearson Correlation Coefficient")
    plt.title("Correlation between Wavelength Profiles and Broadband Mean Profile")
    plt.grid()
    plt.show()


def compare_wavelengths_to_mean_rmse(data, wavelengths, sigma=5, track_index=0):
    """
    Computes and plots the RMSE between each wavelength's profile and the broadband mean.
    The x-axis uses actual wavelength values.
    """
    single_slit = data[track_index, :, :]
    mean_profile = np.mean(single_slit, axis=1)
    smoothed_mean = gaussian_filter1d(mean_profile, sigma=sigma)
    n_wavelengths = single_slit.shape[1]
    rmses = []
    for i in range(n_wavelengths):
        wavelength_profile = single_slit[:, i]
        smoothed_wavelength = gaussian_filter1d(wavelength_profile, sigma=sigma)
        rmse = np.sqrt(np.mean((smoothed_wavelength - smoothed_mean) ** 2))
        rmses.append(rmse)
    plt.figure(figsize=(10, 4))
    plt.bar(
        wavelengths,
        rmses,
        width=(wavelengths[1] - wavelengths[0]) if len(wavelengths) > 1 else 1.0,
        color="steelblue",
    )
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("RMSE")
    plt.title("RMSE between Wavelength Profiles and Broadband Mean Profile")
    plt.show()


def plot_pixel_temporal_evolution(data, pixel_index=200, sigma=5):
    """
    Plots the temporal evolution of the average intensity for a specific slit pixel.
    """
    pixel_intensity = np.mean(data[:, pixel_index, :], axis=1)
    smoothed_intensity = gaussian_filter1d(pixel_intensity, sigma=sigma)
    plt.figure(figsize=(8, 4))
    plt.plot(pixel_intensity, label="Original Intensity")
    plt.plot(
        smoothed_intensity, label=f"Smoothed Intensity (sigma={sigma})", linewidth=2
    )
    plt.xlabel("Track Index (Time Step)")
    plt.ylabel(f"Average Intensity at Slit Pixel {pixel_index}")
    plt.title(f"Temporal Evolution for Slit Pixel {pixel_index}")
    plt.legend()
    plt.show()


def plot_intensity_heatmap(
    data, temporal_index=None, spatial_pixel=None, exclude_ranges=None
):
    intensity_map = np.mean(data, axis=2)

    plt.figure(figsize=(10, 6))
    plt.imshow(intensity_map, aspect="auto", origin="lower", cmap="inferno")
    plt.colorbar(label="Average Intensity (across wavelengths)")

    if temporal_index is not None:
        plt.axhline(y=temporal_index, color="cyan", linestyle="--", linewidth=2)

    if spatial_pixel is not None:
        plt.axvline(x=spatial_pixel, color="lime", linestyle="--", linewidth=2)

    if exclude_ranges:
        for start, end in exclude_ranges:
            plt.axvline(
                x=start, color="white", linestyle="-", linewidth=2
            )  # Left boundary
            plt.axvline(
                x=end, color="white", linestyle="-", linewidth=2
            )  # Right boundary

    plt.xlabel("Slit Pixel Index")
    plt.ylabel("Track Index (Time Step)")
    plt.title("Intensity Heatmap Over Time and Slit Position")
    plt.show()


def plot_global_stats_over_time(data):
    """
    Plots global statistics (mean, median, standard deviation) of the intensity for each track.
    """
    intensity_map = np.mean(data, axis=2)
    mean_vals = np.mean(intensity_map, axis=1)
    median_vals = np.median(intensity_map, axis=1)
    std_vals = np.std(intensity_map, axis=1)
    tracks = np.arange(intensity_map.shape[0])
    plt.figure()
    plt.plot(tracks, mean_vals, label="Mean")
    plt.plot(tracks, median_vals, label="Median")
    plt.plot(tracks, std_vals, label="Std Dev")
    plt.xlabel("Track Index (Time Step)")
    plt.ylabel("Intensity")
    plt.title("Global Statistics Across All Pixels vs. Time")
    plt.legend()
    plt.show()


def plot_mean_tracks(data, skip_interval=5, sigma=0):
    """
    Plots the intensity profiles for every 'skip_interval' track, along with the overall mean profile.
    """
    intensity_map = np.mean(data, axis=2)
    if sigma > 0:
        intensity_map = np.apply_along_axis(
            lambda x: gaussian_filter1d(x, sigma=sigma), axis=1, arr=intensity_map
        )
    mean_profile = np.mean(intensity_map, axis=0)
    plt.figure(figsize=(10, 6))
    for i in range(0, intensity_map.shape[0], skip_interval):
        plt.plot(intensity_map[i, :], color="blue", alpha=0.3)
    plt.plot(mean_profile, color="red", linewidth=2, label="Mean Profile")
    plt.xlabel("Slit Pixel Index")
    plt.ylabel("Intensity")
    plt.title(f"Intensity Profiles (Every {skip_interval} Tracks) with Overall Mean")
    plt.legend()
    plt.show()


def plot_mean_wvls(data, track_index, skip_interval=5):
    wvl_profiles = data[track_index, :, ::skip_interval]

    plt.figure(figsize=(10, 6))

    # Plot each individual wavelength profile
    for i in range(wvl_profiles.shape[1]):
        plt.plot(
            wvl_profiles[:, i],
            alpha=0.3,
        )  # limit legend clutter

    # Compute and plot the mean profile
    mean_profile = np.mean(wvl_profiles, axis=1)
    plt.plot(mean_profile, color="red", linewidth=2, label="Mean Profile")

    plt.xlabel("Slit Pixel Index")
    plt.ylabel("Intensity")
    plt.title(f"Intensity Profiles (Every {skip_interval} wvls) with Overall Mean")
    plt.legend()
    plt.show()


def polynomial_approximate_profile(
    data, track_index=0, poly_degree=3, exclude_ranges=None
):
    """
    Fits a polynomial to the broadband intensity profile for one track,
    excluding specified pixel ranges.
    """
    single_slit = data[track_index, :, :]
    original_profile = np.mean(single_slit, axis=1)
    n_slit_pixels = original_profile.shape[0]
    x = np.arange(n_slit_pixels)
    mask = np.ones(n_slit_pixels, dtype=bool)
    if exclude_ranges is not None:
        for start, end in exclude_ranges:
            mask[start:end] = False
    valid_x = x[mask]
    valid_y = original_profile[mask]
    if len(valid_x) < poly_degree + 1:
        raise ValueError("Not enough valid points to fit the polynomial.")
    coeffs = np.polyfit(valid_x, valid_y, deg=poly_degree)
    fitted_profile = np.polyval(coeffs, x)
    return x, original_profile, fitted_profile


def correct_track_excluded_region(track, poly_degree, exclude_ranges, alpha=1.0):
    """
    Corrects a single track's intensity profile only in the excluded regions.
    """
    profile = np.mean(track, axis=1)
    n_slit_pixels = profile.shape[0]
    x = np.arange(n_slit_pixels)
    include_mask = np.ones(n_slit_pixels, dtype=bool)
    for start, end in exclude_ranges:
        include_mask[start:end] = False
    coeff = np.polyfit(x[include_mask], profile[include_mask], poly_degree)
    poly_fit = np.polyval(coeff, x)
    corrected_track = track.copy()
    for i in range(n_slit_pixels):
        if not include_mask[i]:
            delta = poly_fit[i] - profile[i]
            corrected_track[i, :] = track[i, :] + alpha * delta
    return corrected_track, profile, poly_fit


def plot_corrected_profile_adjustment(
    data, track_index, poly_degree, exclude_ranges, alpha=1.0
):
    """
    Plots the original, fitted, and corrected intensity profiles for one track.
    """
    track = data[track_index, :, :]
    corrected_track, profile, poly_fit = correct_track_excluded_region(
        track, poly_degree, exclude_ranges, alpha
    )
    corrected_profile = np.mean(corrected_track, axis=1)
    x = np.arange(profile.shape[0])
    plt.figure(figsize=(10, 5))
    plt.plot(x, profile, label="Original Profile", color="blue")
    plt.plot(
        x,
        poly_fit,
        label=f"Polynomial Fit (deg={poly_degree})",
        color="red",
        linestyle="--",
    )
    plt.plot(x, corrected_profile, label="Corrected Profile", color="green")
    for start, end in exclude_ranges:
        plt.axvspan(start, end, color="gray", alpha=0.3, label="Excluded Region")
    plt.xlabel("Slit Pixel Index")
    plt.ylabel("Average Intensity")
    plt.title(f"Track {track_index} Profile Correction (alpha={alpha})")
    plt.legend()
    plt.show()


def get_corrected_profile(data, poly_degree, exclude_ranges, alpha=1.0):
    """
    Applies correction to all tracks in the dataset.
    """
    n_tracks = data.shape[0]
    corrected_data = np.empty_like(data)
    for track_index in range(n_tracks):
        track = data[track_index, :, :]
        corrected_track, _, _ = correct_track_excluded_region(
            track, poly_degree, exclude_ranges, alpha
        )
        corrected_data[track_index, :, :] = corrected_track
    return corrected_data


def plot_spatial_spectre(data, track_index=0):
    """
    Plots the spatial intensity profile along the slit for a single track.
    """
    single_slit = np.mean(data[track_index, :, :], axis=1)
    plt.figure(figsize=(8, 4))
    plt.plot(single_slit, label="Intensity Profile")
    plt.xlabel("Slit Pixel (Spatial Position)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.title(f"Spatial Intensity Profile (Track {track_index})")
    plt.show()


def plot_spectral_profile(
    data, track_index, spatial_index, wavelengths, title="Spectral Profile"
):
    """
    Plots the hyperspectral intensity profile for a specific spatial pixel.
    Uses the provided actual wavelengths (in nm) as the x-axis.
    """
    if data.ndim != 3:
        raise ValueError(
            "Input data must be a 3D array (tracks, slit_pixels, wavelengths)."
        )
    spectral_profile = data[track_index, spatial_index, :]
    x_axis = wavelengths
    x_label = "Wavelength (nm)"
    plt.figure(figsize=(8, 4))
    plt.plot(
        x_axis, spectral_profile, label=f"Pixel {spatial_index} at Track {track_index}"
    )
    plt.xlabel(x_label)
    plt.ylabel("Intensity")
    plt.title(title)
    plt.legend()
    plt.show()


def compare_spectral_profiles(
    original_data, corrected_data, track_index, spatial_index, wavelengths
):
    """
    Compares the hyperspectral intensity profiles before and after correction for a specific spatial pixel.
    """
    if original_data.ndim != 3 or corrected_data.ndim != 3:
        raise ValueError(
            "Both inputs must be 3D arrays (tracks, slit_pixels, wavelengths)."
        )
    original_profile = original_data[track_index, spatial_index, :]
    corrected_profile = corrected_data[track_index, spatial_index, :]
    x_axis = wavelengths
    x_label = "Wavelength (nm)"
    plt.figure(figsize=(8, 4))
    plt.plot(
        x_axis, original_profile, label="Original (before correction)", linestyle="-"
    )
    plt.plot(x_axis, corrected_profile, label="Corrected", linestyle="-")
    plt.xlabel(x_label)
    plt.ylabel("Intensity")
    plt.title(f"Spectral Comparison (Track {track_index}, Pixel {spatial_index})")
    plt.legend()
    plt.show()


def correct_illumination_all_tracks(
    data, original_path, poly_degree, exclude_ranges, output_path, smooth_sigma=None
):
    """
    Applies illumination correction to every track and writes the corrected cube to a new H5 file.
    """
    n_tracks, n_slit_pixels, n_wavelengths = data.shape
    corrected_data = np.empty_like(data)
    x = np.arange(n_slit_pixels)
    include_mask = np.ones_like(x, dtype=bool)
    for start, end in exclude_ranges:
        include_mask[start:end] = False
    for track_index in range(n_tracks):
        track = data[track_index, :, :]
        profile = np.mean(track, axis=1)
        if smooth_sigma is not None:
            profile = gaussian_filter1d(profile, sigma=smooth_sigma)
        coeff = np.polyfit(x[include_mask], profile[include_mask], poly_degree)
        poly_fit = np.polyval(coeff, x)
        I_ref = np.mean(profile[include_mask])
        epsilon = 1e-8
        correction_factors = (poly_fit + epsilon) / I_ref
        corrected_track = track * correction_factors[:, np.newaxis]
        corrected_data[track_index, :, :] = corrected_track

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(original_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        for item in f_in:
            f_in.copy(item, f_out, name=item)
        for attr, value in f_in.attrs.items():
            f_out.attrs[attr] = value
        old_ds = f_in["processed/radiance/dataCube"]
        old_attrs = dict(old_ds.attrs)
        old_chunks = old_ds.chunks
        old_compression = old_ds.compression
        old_compression_opts = old_ds.compression_opts
        old_shuffle = old_ds.shuffle
        old_fletcher32 = old_ds.fletcher32
        old_dtype = old_ds.dtype
        del f_out["processed/radiance/dataCube"]
        new_ds = f_out["processed/radiance"].create_dataset(
            "dataCube",
            data=corrected_data,
            dtype=old_dtype,
            chunks=old_chunks,
            compression=old_compression,
            compression_opts=old_compression_opts,
            shuffle=old_shuffle,
            fletcher32=old_fletcher32,
        )
        for k, v in old_attrs.items():
            new_ds.attrs[k] = v

    print(f"Corrected data cube saved to: {output_path}")
    return corrected_data


import os
import h5py


# def save_corrected_data(corrected_data, output_path, original_path=None):
#     """
#     Saves the corrected data cube to an HDF5 file.

#     If an original_path is provided, the function copies all groups and attributes
#     from the original file and replaces the "processed/radiance/dataCube" dataset
#     with the provided corrected_data. If original_path is None, a minimal file with
#     a "processed/radiance" group and "dataCube" dataset is created.

#     Parameters:
#     - corrected_data: NumPy array containing the corrected data.
#     - output_path: Path to save the new HDF5 file.
#     - original_path: (Optional) Path to the original H5 file to copy metadata and structure.
#     """

#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     if original_path is not None:
#         with h5py.File(original_path, "r") as f_in, h5py.File(
#             output_path, "w"
#         ) as f_out:
#             # Copy all groups/datasets and attributes from the original file.
#             for item in f_in:
#                 f_in.copy(item, f_out, name=item)
#             for attr, value in f_in.attrs.items():
#                 f_out.attrs[attr] = value

#             # Retrieve old dataset parameters from the original "dataCube".
#             old_ds = f_in["processed/radiance/dataCube"]
#             old_attrs = dict(old_ds.attrs)
#             old_chunks = old_ds.chunks
#             old_compression = old_ds.compression
#             old_compression_opts = old_ds.compression_opts
#             old_shuffle = old_ds.shuffle
#             old_fletcher32 = old_ds.fletcher32
#             old_dtype = old_ds.dtype

#             # Delete the old dataset and create a new one with the corrected data.
#             del f_out["processed/radiance/dataCube"]
#             new_ds = f_out["processed/radiance"].create_dataset(
#                 "dataCube",
#                 data=corrected_data,
#                 dtype=old_dtype,
#                 chunks=old_chunks,
#                 compression=old_compression,
#                 compression_opts=old_compression_opts,
#                 shuffle=old_shuffle,
#                 fletcher32=old_fletcher32,
#             )
#             for k, v in old_attrs.items():
#                 new_ds.attrs[k] = v
#     else:
#         # If no original file is provided, simply create a new file with a minimal structure.
#         with h5py.File(output_path, "w") as f_out:
#             group = f_out.create_group("processed/radiance")
#             group.create_dataset("dataCube", data=corrected_data)

#     print(f"Corrected data cube saved to: {output_path}")
import os
import h5py


def save_corrected_data(corrected_data, original_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(original_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        # Copy all groups/datasets and attributes from the original file
        for item in f_in:
            f_in.copy(item, f_out, name=item)
        for attr, value in f_in.attrs.items():
            f_out.attrs[attr] = value

        old_ds = f_in["processed/radiance/dataCube"]

        # If the shape differs (e.g. due to shortening), simply create a new dataset
        if corrected_data.shape != old_ds.shape:
            del f_out["processed/radiance/dataCube"]
            new_ds = f_out["processed/radiance"].create_dataset(
                "dataCube", data=corrected_data
            )
            print("bobbob")
        else:
            old_attrs = dict(old_ds.attrs)
            old_chunks = old_ds.chunks
            old_compression = old_ds.compression
            old_compression_opts = old_ds.compression_opts
            old_shuffle = old_ds.shuffle
            old_fletcher32 = old_ds.fletcher32
            old_dtype = old_ds.dtype

            del f_out["processed/radiance/dataCube"]
            new_ds = f_out["processed/radiance"].create_dataset(
                "dataCube",
                data=corrected_data,
                dtype=old_dtype,
                chunks=old_chunks,
                compression=old_compression,
                compression_opts=old_compression_opts,
                shuffle=old_shuffle,
                fletcher32=old_fletcher32,
            )
            for k, v in old_attrs.items():
                new_ds.attrs[k] = v

    print(f"Corrected data cube saved to: {output_path}")


import os
import h5py


def save_shortened_data(shortened_data, original_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)
    with h5py.File(original_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        # Copy file-level attributes
        for attr, value in f_in.attrs.items():
            f_out.attrs[attr] = value

        # Loop over top-level keys
        for key in f_in.keys():
            if key != "processed":
                f_in.copy(key, f_out, name=key)
            else:
                # Create the "processed" group in the output file
                proc_in = f_in["processed"]
                proc_out = f_out.create_group("processed")
                # Loop over subgroups in "processed"
                for subkey in proc_in.keys():
                    if subkey != "radiance":
                        proc_in.copy(subkey, proc_out, name=subkey)
                    else:
                        # Create the "radiance" group and copy everything except "dataCube"
                        rad_in = proc_in["radiance"]
                        rad_out = proc_out.create_group("radiance")
                        for item in rad_in.keys():
                            if item != "dataCube":
                                rad_in.copy(item, rad_out, name=item)
                        # Create new "dataCube" using the shortened data
                        ds_in = rad_in["dataCube"]
                        ds_attrs = dict(ds_in.attrs)
                        new_ds = rad_out.create_dataset(
                            "dataCube",
                            data=shortened_data,
                            dtype=ds_in.dtype,
                            chunks=ds_in.chunks,
                            compression=ds_in.compression,
                            compression_opts=ds_in.compression_opts,
                            shuffle=ds_in.shuffle,
                            fletcher32=ds_in.fletcher32,
                        )
                        for k, v in ds_attrs.items():
                            new_ds.attrs[k] = v
    print("Shortened file saved to:", output_path)


############################### SPLINE FITTING ###############################
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt


def correct_track_excluded_region_spline(
    track, exclude_ranges, smoothing_factor=1000.0, alpha=1.0
):
    """
    Corrects a single track's intensity profile using a smoothing spline, ignoring excluded regions.

    Parameters:
    -----------
    track : np.ndarray, shape (n_slit_pixels, n_wavelengths)
        The hyperspectral data for one 'track' (time step).
    exclude_ranges : list of (start, end) tuples
        Pixel index ranges to exclude from the spline fit.
    smoothing_factor : float
        Smoothing parameter for the spline (UnivariateSpline 's'). Higher means smoother fit.
    alpha : float
        How strongly to apply the correction in the excluded regions.
        If alpha=1, fully correct to match the spline. If alpha<1, partially correct.

    Returns:
    --------
    corrected_track : np.ndarray
        A copy of 'track' but with corrected intensities in the excluded regions.
    original_profile : np.ndarray
        The mean intensity profile along the slit before correction (shape: (n_slit_pixels,)).
    spline_fit : np.ndarray
        The fitted spline values at each slit pixel (same shape).
    """
    # 1) Compute the original mean profile along the slit
    original_profile = np.mean(track, axis=1)  # shape: (n_slit_pixels,)
    n_slit_pixels = original_profile.size
    x = np.arange(n_slit_pixels)

    # 2) Create a mask for the "good" (included) regions
    include_mask = np.ones(n_slit_pixels, dtype=bool)
    if exclude_ranges is not None:
        for start, end in exclude_ranges:
            include_mask[start:end] = False

    # 3) Fit a smoothing spline only on the included data
    #    (UnivariateSpline will not automatically exclude points, so we do it manually)
    x_included = x[include_mask]
    y_included = original_profile[include_mask]

    # Make sure we have enough points to fit:
    if len(x_included) < 4:
        raise ValueError("Not enough valid (included) points to fit the spline.")

    spline = UnivariateSpline(x_included, y_included, s=smoothing_factor)
    spline_fit = spline(x)  # Evaluate spline at all x

    # 4) Apply correction ONLY in the excluded regions
    corrected_track = track.copy()
    for i in range(n_slit_pixels):
        if not include_mask[i]:
            # "Difference" approach: raise or lower the intensities by (spline - original)
            delta = spline_fit[i] - original_profile[i]
            corrected_track[i, :] = track[i, :] + alpha * delta

    return corrected_track, original_profile, spline_fit


def plot_corrected_profile_adjustment_spline(
    data, track_index, exclude_ranges, smoothing_factor=1000.0, alpha=1.0
):
    """
    Plots the original, spline-fitted, and corrected intensity profiles for one track,
    ignoring specified pixel ranges during spline fitting.

    Parameters:
    -----------
    data : np.ndarray, shape (n_tracks, n_slit_pixels, n_wavelengths)
        The entire hyperspectral dataset.
    track_index : int
        Which track (time step) to correct/plot.
    exclude_ranges : list of (start, end) tuples
        Pixel index ranges to exclude from spline fitting.
    smoothing_factor : float
        Smoothing parameter for UnivariateSpline.
    alpha : float
        Strength of the correction (1.0 = full).
    """
    # Extract the single track data
    track = data[track_index, :, :]  # shape: (n_slit_pixels, n_wavelengths)

    # Correct it via spline
    corrected_track, original_profile, spline_fit = (
        correct_track_excluded_region_spline(
            track, exclude_ranges, smoothing_factor, alpha
        )
    )

    # Compare the profiles
    corrected_profile = np.mean(corrected_track, axis=1)
    x = np.arange(len(original_profile))

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, original_profile, label="Original Profile", color="blue")
    plt.plot(x, spline_fit, label=f"Spline Fit", color="red", linestyle="--")
    plt.plot(x, corrected_profile, label="Corrected Profile", color="green")
    for start, end in exclude_ranges:
        plt.axvspan(start, end, color="gray", alpha=0.3)
    plt.xlabel("Slit Pixel Index")
    plt.ylabel("Average Intensity")
    plt.title(f"Track {track_index} Profile Correction (Spline, alpha={alpha})")
    plt.legend()
    plt.show()


def get_corrected_profile_spline(
    data, exclude_ranges, smoothing_factor=1000.0, alpha=1.0
):
    """
    Applies spline-based illumination correction to ALL tracks in the dataset.

    Parameters:
    -----------
    data : np.ndarray, shape (n_tracks, n_slit_pixels, n_wavelengths)
        The entire hyperspectral dataset.
    exclude_ranges : list of (start, end) tuples
        Pixel index ranges to exclude from spline fitting.
    smoothing_factor : float
        Smoothing parameter for UnivariateSpline.
    alpha : float
        Strength of correction in excluded regions.

    Returns:
    --------
    corrected_data : np.ndarray
        The same shape as 'data' but corrected track-by-track.
    """
    n_tracks = data.shape[0]
    corrected_data = np.empty_like(data)
    for track_idx in range(n_tracks):
        track = data[track_idx, :, :]
        corrected_track, _, _ = correct_track_excluded_region_spline(
            track, exclude_ranges, smoothing_factor, alpha
        )
        corrected_data[track_idx, :, :] = corrected_track
    return corrected_data


import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt


# todo
def correct_track_spline_full(track, exclude_ranges, smoothing_factor=1000.0):
    """
    Corrects a single track's intensity profile so that the *mean* along the slit
    matches a smoothing spline fit, using an additive correction.
    This preserves the original spectral fluctuations (noise) within each slit pixel by
    computing an offset from the spline fit and applying it to the entire spectral data.

    Parameters
    ----------
    track : np.ndarray, shape (n_slit_pixels, n_wavelengths)
        Hyperspectral data for one track (time step).
    exclude_ranges : list of (start, end) tuples
        Slit pixel index ranges to exclude from the spline fit (where the data is obviously bad).
    smoothing_factor : float
        Smoothing parameter for UnivariateSpline. Larger values yield a smoother spline.

    Returns
    -------
    corrected_track : np.ndarray, shape (n_slit_pixels, n_wavelengths)
        The fully corrected track data.
    original_profile : np.ndarray, shape (n_slit_pixels,)
        The original average (over wavelengths) intensity profile.
    spline_fit : np.ndarray, shape (n_slit_pixels,)
        The spline-fitted profile for each slit pixel.
    offset : np.ndarray, shape (n_slit_pixels,)
        The additive offset (spline_fit - original_profile) applied to each slit pixel.
    """
    # 1) Compute the original mean profile across wavelengths

    # track = data[track_index, :, :]  # shape: (n_slit_pixels, n_wavelengths)
    #
    original_profile = np.mean(track, axis=1)  # shape: (n_slit_pixels,)
    n_slit_pixels = len(original_profile)
    x = np.arange(n_slit_pixels)

    # 2) Create a mask to exclude the "bad" regions
    include_mask = np.ones(n_slit_pixels, dtype=bool)
    if exclude_ranges:
        for start, end in exclude_ranges:
            include_mask[start:end] = False

    # 3) Fit a smoothing spline to the "good" data only
    x_included = x[include_mask]
    y_included = original_profile[include_mask]
    spline = UnivariateSpline(x_included, y_included, s=smoothing_factor)
    spline_fit = spline(x)  # Evaluate spline at all x

    # 4) Compute the additive offset: offset = spline_fit - original_profile
    offset = spline_fit - original_profile

    # 5) Apply this offset to *all wavelengths* for each pixel
    corrected_track = track + offset[:, np.newaxis]

    return corrected_track, original_profile, spline_fit, offset


########## THIS IS FIRST FOR PLOTTING ################
def plot_corrected_profile_adjustment_spline_full(
    data, track_index, exclude_ranges=None, smoothing_factor=1000.0
):
    """
    Plots the original average profile, the spline fit, and the corrected average profile
    for one track using the additive spline correction.

    Parameters
    ----------
    data : np.ndarray, shape (n_tracks, n_slit_pixels, n_wavelengths)
        The hyperspectral dataset.
    track_index : int
        Which track (time step) to correct/plot.
    exclude_ranges : list of (start, end) tuples, optional
        Pixel index ranges to exclude from spline fitting.
    smoothing_factor : float, optional
        Smoothing parameter for UnivariateSpline.
    """
    track = data[track_index, :, :]  # shape: (n_slit_pixels, n_wavelengths)

    # Correct using the additive offset approach
    corrected_track, original_profile, spline_fit, offset = correct_track_spline_full(
        track, exclude_ranges=exclude_ranges, smoothing_factor=smoothing_factor
    )

    corrected_profile = np.mean(corrected_track, axis=1)
    x = np.arange(len(original_profile))

    plt.figure(figsize=(10, 5))
    plt.plot(x, original_profile, label="Original Mean Profile", color="blue")
    plt.plot(x, spline_fit, label="Spline Fit", color="red", linestyle="--")
    plt.plot(x, corrected_profile, label="Corrected Mean Profile", color="green")

    if exclude_ranges:
        for start, end in exclude_ranges:
            plt.axvspan(start, end, color="gray", alpha=0.2)

    plt.xlabel("Slit Pixel Index")
    plt.ylabel("Average Intensity")
    plt.title(f"Track {track_index} Profile Correction (Spline, additive correction)")
    plt.legend()
    plt.show()


def get_corrected_profile_spline_full(
    data, exclude_ranges=None, smoothing_factor=1000.0
):
    """
    Applies the ratio-based spline correction to *all* tracks in the dataset.

    Parameters
    ----------
    data : np.ndarray, shape (n_tracks, n_slit_pixels, n_wavelengths)
        The entire hyperspectral dataset.
    exclude_ranges : list of (start, end) tuples
        Pixel index ranges to exclude from the spline fit (same for all tracks).
    smoothing_factor : float
        Smoothing parameter for UnivariateSpline.

    Returns
    -------
    corrected_data : np.ndarray, shape (n_tracks, n_slit_pixels, n_wavelengths)
        The fully corrected dataset.
    """
    n_tracks = data.shape[0]
    corrected_data = np.empty_like(data)
    for t in range(n_tracks):
        track = data[t, :, :]
        corrected_track, _, _, _ = correct_track_spline_full(
            track, exclude_ranges=exclude_ranges, smoothing_factor=smoothing_factor
        )
        corrected_data[t, :, :] = corrected_track
    return corrected_data


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


def plot_corrected_profile_adjustment_spline_full_with_local_variations(
    data,
    track_index,
    exclude_ranges=None,
    global_smoothing_factor=1000.0,
    local_smoothing_factor=50.0,
):
    """
    Plots the original average profile, the global spline fit,
    and the corrected average profile for one track.
    We preserve local variations by also fitting a "local" spline
    to the original profile and adding those small wiggles back
    onto the global baseline.

    Parameters
    ----------
    data : np.ndarray, shape (n_tracks, n_slit_pixels, n_wavelengths)
    track_index : int
        Which track (time step) to correct/plot.
    exclude_ranges : list of (start, end) tuples
        Pixel index ranges to exclude from spline fitting.
    global_smoothing_factor : float
        Smoothing parameter for the "global" UnivariateSpline.
        Larger => smoother baseline.
    local_smoothing_factor : float
        Smoothing parameter for the "local" UnivariateSpline
        that extracts smaller-scale shape from the original profile.
    """
    # Extract the track
    track = data[track_index, :, :]  # shape: (n_slit_pixels, n_wavelengths)

    # Correct using the two-spline approach
    (
        corrected_track,
        original_profile,
        corrected_profile,
        global_spline_fit,
        local_spline_fit,
    ) = correct_track_spline_full_with_local_variations(
        track,
        exclude_ranges=exclude_ranges,
        global_smoothing_factor=global_smoothing_factor,
        local_smoothing_factor=local_smoothing_factor,
    )

    # Prepare for plotting
    x = np.arange(len(original_profile))
    plt.figure(figsize=(10, 5))

    plt.plot(x, original_profile, label="Original Mean Profile", color="blue")
    plt.plot(
        x, global_spline_fit, label="Global Spline Fit", color="red", linestyle="--"
    )
    plt.plot(
        x,
        corrected_profile,
        label="Corrected Mean (Preserve Local Variations)",
        color="green",
    )

    # (Optional) also plot the local spline if you want to see it
    plt.plot(
        x,
        local_spline_fit,
        label="Local Spline (extracted baseline)",
        color="magenta",
        linestyle=":",
    )

    # Highlight excluded regions
    if exclude_ranges:
        for start, end in exclude_ranges:
            plt.axvspan(start, end, color="gray", alpha=0.2)

    plt.xlabel("Slit Pixel Index")
    plt.ylabel("Average Intensity")
    plt.title(f"Track {track_index} Profile Correction with Local Variations Preserved")
    plt.legend()
    plt.show()


def correct_track_spline_full_with_local_variations(
    track,
    exclude_ranges=None,
    global_smoothing_factor=1000.0,
    local_smoothing_factor=50.0,
):
    """
    Corrects a single track's intensity profile so that the "global" mean along the slit
    matches a smoothing spline fit (the global baseline) but retains local variations
    from the original data. We do this by:

    1) Fitting a global spline to the original mean (excluding bad ranges) to define
       the new baseline we trust.
    2) Fitting a 'local' spline (less smoothed) to the original mean to capture
       the broad shape that we want to treat as 'baseline' for local wiggles.
    3) Subtracting the local spline from the original mean => local_wiggles
    4) Adding local_wiggles to the global spline => corrected mean
    5) Applying that pixel-by-pixel offset to the entire hyperspectral track.

    Parameters
    ----------
    track : np.ndarray, shape (n_slit_pixels, n_wavelengths)
        Hyperspectral data for one track (time step).
    exclude_ranges : list of (start, end) tuples
        Pixel index ranges to exclude from the global spline fit.
    global_smoothing_factor : float
        Smoothing parameter for the global UnivariateSpline.
        Larger => smoother baseline.
    local_smoothing_factor : float
        Smoothing parameter for the local UnivariateSpline.
        Smaller => it follows local features more closely.

    Returns
    -------
    corrected_track : np.ndarray, shape (n_slit_pixels, n_wavelengths)
        The fully corrected track data.
    original_profile : np.ndarray, shape (n_slit_pixels,)
        The original average intensity profile.
    corrected_profile : np.ndarray, shape (n_slit_pixels,)
        The new corrected mean profile (global baseline + local wiggles).
    global_spline_fit : np.ndarray, shape (n_slit_pixels,)
        The "global" spline baseline for each slit pixel.
    local_spline_fit : np.ndarray, shape (n_slit_pixels,)
        The "local" spline fit to the original mean (used to extract wiggles).
    """
    # 1) Compute the original mean profile across wavelengths
    original_profile = np.mean(track, axis=1)
    n_slit_pixels = len(original_profile)
    x = np.arange(n_slit_pixels)

    # 2) Create a mask to exclude the "bad" regions from the *global* fit
    include_mask = np.ones(n_slit_pixels, dtype=bool)
    if exclude_ranges:
        for start, end in exclude_ranges:
            include_mask[start:end] = False

    # 3) Fit the "global" smoothing spline (large smoothing) to define the baseline
    x_included = x[include_mask]
    y_included = original_profile[include_mask]
    global_spline = UnivariateSpline(x_included, y_included, s=global_smoothing_factor)
    global_spline_fit = global_spline(x)

    # 4) Fit a "local" spline (less smoothing) on the *same data*
    #    This local spline is used to figure out the broad shape in the original
    #    so we can isolate the smaller wiggles.
    local_spline = UnivariateSpline(x_included, y_included, s=local_smoothing_factor)
    local_spline_fit = local_spline(x)

    # 5) local_wiggles = original_profile - local_spline_fit
    local_wiggles = original_profile - local_spline_fit

    # 6) The new corrected mean is:
    #    corrected_profile = global baseline + local wiggles
    corrected_profile = global_spline_fit + local_wiggles

    # 7) Convert that new mean profile into a pixel-by-pixel offset
    offset = corrected_profile - original_profile  # shape: (n_slit_pixels,)

    # 8) Apply this offset to *all wavelengths* for each pixel
    corrected_track = track + offset[:, np.newaxis]

    return (
        corrected_track,
        original_profile,
        corrected_profile,
        global_spline_fit,
        local_spline_fit,
    )


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def plot_selected_wavelength_profiles(
    data, wavelengths, track_index=0, selected_wavelengths=None, sigma=5
):
    """
    Plots the spatial intensity profile along the slit for one track.
    The plot includes:
      - The broadband mean intensity (average over all wavelengths)
      - Intensity profiles at selected wavelengths (using interpolation if necessary)

    Parameters
    ----------
    data : np.ndarray
        3D array with shape (n_tracks, n_slit_pixels, n_wavelengths)
    wavelengths : np.ndarray or list
        1D array of actual wavelength values corresponding to the spectral axis.
    track_index : int
        Which track (time step) to plot.
    selected_wavelengths : list or np.ndarray, optional
        Wavelength values (in nm) at which to extract and plot intensity profiles.
        If None, the function will automatically choose values spaced about 100 nm apart
        over the range of 'wavelengths'.
    sigma : float, optional
        Standard deviation for Gaussian smoothing (applied to both the broadband and
        selected-wavelength profiles). Use sigma=0 for no smoothing.

    Returns
    -------
    None. The function creates a matplotlib plot.
    """
    # Extract the data for the specified track.
    # track shape: (n_slit_pixels, n_wavelengths)
    track = data[track_index, :, :]
    n_slit_pixels = track.shape[0]
    x = np.arange(n_slit_pixels)

    # Compute the broadband mean (averaging over all wavelengths)
    mean_profile = np.mean(track, axis=1)
    if sigma > 0:
        mean_profile = gaussian_filter1d(mean_profile, sigma=sigma)

    # If no selected wavelengths are provided, choose them automatically.
    if selected_wavelengths is None:
        min_wl, max_wl = np.min(wavelengths), np.max(wavelengths)
        # Create a range spaced by ~100 nm.
        selected_wavelengths = np.arange(np.ceil(min_wl / 100) * 100, max_wl, 100)

    # Create a figure for the plot
    plt.figure(figsize=(10, 6))

    # Plot the broadband mean profile
    plt.plot(x, mean_profile, label="Broadband Mean", color="black", linewidth=2)

    # For each selected wavelength, interpolate its intensity profile along the slit
    for wl in selected_wavelengths:
        # For each slit pixel, interpolate intensity at this wavelength.
        profile = np.array(
            [np.interp(wl, wavelengths, track[i, :]) for i in range(n_slit_pixels)]
        )
        if sigma > 0:
            profile = gaussian_filter1d(profile, sigma=sigma)
        plt.plot(x, profile, label=f"{wl:.0f} nm", linestyle="--")

    plt.xlabel("Slit Pixel (Spatial Position)")
    plt.ylabel("Intensity")
    plt.title(
        f"Spatial Intensity Profiles (Track {track_index})\nBroadband Mean and Selected Wavelengths"
    )
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage:
# Suppose you have:
# data: a 3D numpy array of shape (n_tracks, n_slit_pixels, n_wavelengths)
# wavelengths: a 1D array of wavelength values (length = n_wavelengths)
#
# plot_selected_wavelength_profiles(data, wavelengths, track_index=0)
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def plot_normalized_wavelength_profiles(
    data, wavelengths, track_index=0, selected_wavelengths=None, sigma=5
):

    # Extract the data for the specified track
    track = data[track_index, :, :]  # shape: (n_slit_pixels, n_wavelengths)
    n_slit_pixels = track.shape[0]
    x = np.arange(n_slit_pixels)

    # Choose selected wavelengths if not provided
    if selected_wavelengths is None:
        min_wl, max_wl = 450, 700
        selected_wavelengths = np.arange(np.ceil(min_wl / 75) * 75, max_wl, 75)

    plt.figure(figsize=(10, 6))

    # Store all profiles for mean computation
    all_profiles = []

    # For each selected wavelength, interpolate its profile and normalize it
    for wl in selected_wavelengths:
        profile = np.array(
            [np.interp(wl, wavelengths, track[i, :]) for i in range(n_slit_pixels)]
        )
        if sigma > 0:
            profile = gaussian_filter1d(profile, sigma=sigma)
        normalized_profile = (profile - np.min(profile)) / (
            np.max(profile) - np.min(profile)
        )
        plt.plot(x, normalized_profile, label=f"{wl:.0f} nm", linestyle="--")
        all_profiles.append(normalized_profile)

    # Compute and plot the mean profile
    mean_profile = np.mean(all_profiles, axis=0)
    plt.plot(x, mean_profile, label="Mean Profile", linewidth=2, color="black")

    plt.xlabel("Slit Pixel (Spatial Position)")
    plt.ylabel("Normalized Intensity")
    plt.title(f"Normalized Spatial Intensity Profiles (Track {track_index})")
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage:
# plot_normalized_wavelength_profiles(data, wavelengths, track_index=0)


######################## own code ############################


def find_optimal_smoothing(
    profile, baseline, mask_include, candidates=None, error_threshold=None
):
    # i want to test many different smoothing factors, which i here call candidates
    if candidates is None:
        candidates = np.arange(0.01, 1, 0.001)
    elif isinstance(candidates, (int, float)):
        candidates = [candidates]  # Convert single value to a list

    # sf = smooth factor -> determines how smooth the spline is
    best_sf, best_error = None, np.inf

    # x is just the spatial pixel indices
    x = np.arange(baseline.size)

    # error_threshold = 0.001

    for sf in candidates:
        spline = UnivariateSpline(x[mask_include], profile[mask_include], s=sf)(x)
        mse = np.mean((baseline[mask_include] - spline[mask_include]) ** 2)
        if mse < best_error:
            best_error, best_sf = mse, sf
        if error_threshold is not None and mse <= error_threshold:
            break
    print(best_sf)
    print(best_error)
    print("-----------------")
    return best_sf


from scipy.optimize import minimize_scalar


def find_optimal_smoothing_optimization(profile, baseline, mask_include):
    """
    Uses a bounded optimization routine to find the optimal smoothing factor.
    """
    x = np.arange(baseline.size)

    def objective(sf):
        spline = UnivariateSpline(x[mask_include], profile[mask_include], s=sf)(x)
        mse = np.mean((baseline[mask_include] - spline[mask_include]) ** 2)
        return mse

    # Use a bounded method to limit the search between 0.01 and 1
    result = minimize_scalar(objective, bounds=(0.01, 1), method="bounded")
    best_sf = result.x
    best_error = result.fun

    # print(f"Optimal smoothing factor (optimization): {best_sf}, error: {best_error}")
    return best_sf


# takes in single wvl index, and single timestep. we use wvl index since these are known values, while if we use wvl nm its already interpolated
def correct_light(
    data,
    temporal_index,
    wvl_index,
    exclude_ranges=None,
    sigma=20,
    smoothing_factor=None,
):
    # todo
    profile = data[temporal_index, :, wvl_index]
    n_spatial_pixels = profile.shape[0]
    x = np.arange(n_spatial_pixels)

    mask_include = np.ones(n_spatial_pixels, dtype=bool)
    if exclude_ranges is not None:
        for start, end in exclude_ranges:
            mask_include[start:end] = False

    # shall be the bsaeline
    baseline = gaussian_filter1d(profile, sigma)
    offset = profile - baseline

    # finds the optimal smoothing factor
    # best_sf = find_optimal_smoothing(
    #     profile, baseline, mask_include, candidates=smoothing_factor
    # )
    best_sf = find_optimal_smoothing_optimization(profile, baseline, mask_include)
    spline_exl_regions = UnivariateSpline(
        x[mask_include], profile[mask_include], s=best_sf
    )(x)

    # corrected_profile = profile.copy()
    corrected_profile = spline_exl_regions + offset
    return corrected_profile, profile, spline_exl_regions, baseline


def plot_correct_light(
    profile,
    baseline,
    corrected_profile,
    spline_exl_regions,
    temporal_index,
    band_2_wavelength,
    wvl_index,
    exclude_ranges=None,
):
    # 1) Create the x-axis (spatial pixel indices)
    x = np.arange(len(profile))

    # 2) Plot everything
    plt.figure(figsize=(14, 10))
    plt.plot(x, profile, label="Original Profile", color="blue", alpha=0.7)
    plt.plot(x, baseline, label="Baseline", color="purple")
    plt.plot(x, corrected_profile, label="Corrected Profile", color="green")
    plt.plot(x, spline_exl_regions, label="Spline Fit", color="#b8860b")

    # Highlight excluded regions if provided
    if exclude_ranges is not None:
        for start, end in exclude_ranges:
            plt.axvspan(start, end, color="gray", alpha=0.2)

    # 3) Labels and title
    wavelength_nm = round(
        wvl_index_to_nm(wvl_index, band_2_wavelength=band_2_wavelength), 1
    )
    plt.xlabel("Slit Pixel Index")
    plt.ylabel("Intensity")
    plt.title(
        f"Temporal Index: {temporal_index}\nWavelength Index: {wvl_index}\nWavelength [nm]: {wavelength_nm}\n"
        "Profile Correction (Single Wavelength)"
    )

    plt.legend()
    plt.show()


# def correct_intensity_for_each_wavelength(
#     data, exclude_ranges=None, smoothing_factor=0):
#         for i in range()


def get_corrected_data(data, exclude_ranges=None, smoothing_factor=None):
    """
    Applies the correction (via correct_light) for each time step and wavelength,
    returning a corrected data array with the same shape as the input.
    """
    estimate_get_corrected_data_runtime(
        data, exclude_ranges, smoothing_factor, iterations=1
    )

    n_time, _, n_wavelengths = data.shape
    corrected_data = np.empty_like(data)

    for time_index in range(n_time):
        for wvl_index in range(n_wavelengths):
            corrected_profile, _, _, _ = correct_light(
                data,
                time_index,
                wvl_index,
                exclude_ranges,
                smoothing_factor=smoothing_factor,
            )
            corrected_data[time_index, :, wvl_index] = corrected_profile

    return corrected_data


import time
from datetime import datetime


import time
from datetime import datetime

import time
from datetime import datetime, timedelta, timezone


def estimate_get_corrected_data_runtime(
    data, exclude_ranges=None, smoothing_factor=None, iterations=1
):
    n_time, _, n_wavelengths = data.shape
    total_time = 0.0
    iterations = min(iterations, n_time)

    for time_index in range(iterations):
        start = time.time()
        for wvl_index in range(n_wavelengths):
            _ = correct_light(
                data,
                time_index,
                wvl_index,
                exclude_ranges,
                smoothing_factor=smoothing_factor,
            )
        elapsed = time.time() - start
        total_time += elapsed
        # print(f"Time index {time_index+1}/{iterations}: {elapsed:.2f} sec")

    avg_time = total_time / iterations
    estimated_total_time = avg_time * n_time
    print(f"Estimated total runtime: (~{estimated_total_time/60:.2f} min)")

    # Define Norway timezone as UTC+1
    norway_tz = timezone(timedelta(hours=1))
    start_time = datetime.now(norway_tz)
    print("starttime (UTC+1) = " + start_time.strftime("%H:%M"))

    estimated_endtime = start_time + timedelta(seconds=estimated_total_time)
    print("endtime   (UTC+1) = " + estimated_endtime.strftime("%H:%M"))


def normalize_profile(profile):
    """Normalize profile to [0, 1] and return normalization parameters."""
    p_min = profile.min()
    p_max = profile.max()
    norm_profile = (profile - p_min) / (p_max - p_min)
    return norm_profile, p_min, p_max


def denormalize_profile(norm_profile, p_min, p_max):
    """Rescale the normalized profile back to the original range."""
    return norm_profile * (p_max - p_min) + p_min


def correct_light_normalized(
    data, temporal_index, wvl_index, exclude_ranges=None, smoothing_factor=0, sigma=2
):
    # Extract the profile for the given temporal and wavelength indices.
    profile = data[temporal_index, :, wvl_index]

    # Normalize the profile.
    profile_norm, p_min, p_max = normalize_profile(profile)

    n_spatial_pixels = profile_norm.shape[0]
    x = np.arange(n_spatial_pixels)

    # Create an inclusion mask based on the exclude_ranges.
    mask_include = np.ones(n_spatial_pixels, dtype=bool)
    if exclude_ranges is not None:
        for start, end in exclude_ranges:
            mask_include[start:end] = False

    # Apply the spline on the normalized profile.
    spline_norm = UnivariateSpline(
        x[mask_include],
        profile_norm[mask_include],
        s=smoothing_factor,
    )(x)

    # Compute a baseline using a Gaussian filter on the normalized profile.
    baseline_norm = gaussian_filter1d(profile_norm, sigma)
    offset_norm = profile_norm - baseline_norm

    # Combine the spline fit with the offset to get the corrected normalized profile.
    corrected_norm = spline_norm + offset_norm

    # Re-scale the corrected profile back to the original intensity range.
    corrected_profile = denormalize_profile(corrected_norm, p_min, p_max)

    # Optionally, also compute the original spline and baseline in the original scale,
    # if you want to compare them directly.
    spline_orig = denormalize_profile(spline_norm, p_min, p_max)
    baseline_orig = denormalize_profile(baseline_norm, p_min, p_max)

    return corrected_profile, profile, spline_orig, baseline_orig


def plot_correct_light_normalized(
    data, temporal_index, wvl_index, exclude_ranges=None, smoothing_factor=0, sigma=1
):
    # 1) Use the normalized correction function to get profiles.
    corrected_profile, profile, spline, baseline = correct_light_normalized(
        data,
        temporal_index,
        wvl_index,
        exclude_ranges=exclude_ranges,
        smoothing_factor=smoothing_factor,
        sigma=sigma,
    )

    # 2) Create the x-axis (spatial pixel indices)
    x = np.arange(len(profile))

    # 3) Plot everything
    plt.figure(figsize=(8, 5))
    plt.plot(x, profile, label="Original Profile", color="blue", alpha=0.7)
    plt.plot(x, baseline, label="Baseline", color="purple")
    plt.plot(
        x, corrected_profile, label="Corrected Profile", color="green", linestyle=":"
    )
    plt.plot(x, spline, label="Spline Fit", color="#b8860b")

    # Highlight excluded regions if provided
    if exclude_ranges is not None:
        for start, end in exclude_ranges:
            plt.axvspan(start, end, color="gray", alpha=0.2)

    # 4) Labels and title
    plt.xlabel("Slit Pixel Index")
    plt.ylabel("Intensity")
    plt.title(
        f"Temporal Index: {temporal_index}, Wavelength Index: {wvl_index}\n"
        "Profile Correction (Normalized Spline Correction)"
    )

    plt.legend()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt


def correct_light_poly(
    data, temporal_index, wvl_index, exclude_ranges=None, poly_degree=2, alpha=1.0
):

    # Extract the 1D profile for the specified time and wavelength.
    profile = data[temporal_index, :, wvl_index]
    n_spatial_pixels = profile.shape[0]
    x = np.arange(n_spatial_pixels)

    # Create a boolean mask for the included regions.
    mask_include = np.ones(n_spatial_pixels, dtype=bool)
    if exclude_ranges is not None:
        for start, end in exclude_ranges:
            mask_include[start:end] = False

    # Fit a polynomial only to the included points.
    coeff = np.polyfit(x[mask_include], profile[mask_include], poly_degree)
    poly_fit = np.polyval(coeff, x)

    # Copy the profile and then correct only the excluded regions.
    corrected_profile = profile.copy()
    for i in range(n_spatial_pixels):
        if not mask_include[i]:
            delta = poly_fit[i] - profile[i]
            corrected_profile[i] = profile[i] + alpha * delta

    return corrected_profile, profile, poly_fit


def plot_correct_light_poly(
    data, temporal_index, wvl_index, exclude_ranges=None, poly_degree=2, alpha=1.0
):
    """
    Plots the original profile, the polynomial fit, and the corrected profile for a single wavelength
    at a given time step.
    """
    corrected_profile, profile, poly_fit = correct_light_poly(
        data, temporal_index, wvl_index, exclude_ranges, poly_degree, alpha
    )

    x = np.arange(len(profile))
    plt.figure(figsize=(8, 5))
    plt.plot(x, profile, label="Original Profile", color="blue", alpha=0.7)
    plt.plot(x, corrected_profile, label="Corrected Profile", color="green")
    plt.plot(x, poly_fit, label="Polynomial Fit", color="#b8860b")

    # Highlight excluded regions if provided.
    if exclude_ranges is not None:
        for start, end in exclude_ranges:
            plt.axvspan(start, end, color="gray", alpha=0.2)

    plt.xlabel("Slit Pixel Index")
    plt.ylabel("Intensity")
    plt.title(
        f"Temporal Index: {temporal_index}, Wavelength Index: {wvl_index}\nPolynomial Correction (Degree {poly_degree})"
    )
    plt.legend()
    plt.show()


def get_corrected_data_poly(data, exclude_ranges=None, poly_degree=2, alpha=1.0):
    """
    Applies the polynomial-based correction for each time step and wavelength,
    returning a corrected data array with the same shape as the input.
    """
    n_time, n_slit_pixels, n_wavelengths = data.shape
    corrected_data = np.empty_like(data)

    for time_index in range(n_time):
        for wvl_index in range(n_wavelengths):
            corrected_profile, _, _ = correct_light_poly(
                data, time_index, wvl_index, exclude_ranges, poly_degree, alpha
            )
            corrected_data[time_index, :, wvl_index] = corrected_profile

    return corrected_data


def wvl_index_to_nm(wvl_index, band_2_wavelength):
    """
    Convert a wavelength index to the corresponding wavelength value (in nm).
    """
    if wvl_index < 0 or wvl_index >= len(band_2_wavelength):
        raise ValueError(
            f"Index {wvl_index} is out of bounds. Valid range: 0 to {len(band_2_wavelength) - 1}"
        )

    return round(band_2_wavelength[wvl_index])


def wvl_nm_to_index(wvl_nm, band_2_wavelength):

    band_2_wavelength = np.array(band_2_wavelength)  # Ensure it's a NumPy array

    if wvl_nm < band_2_wavelength[0] or wvl_nm > band_2_wavelength[-1]:
        raise ValueError(
            f"Wavelength {wvl_nm} nm is out of range ({band_2_wavelength[0]} - {band_2_wavelength[-1]} nm)"
        )

    # Find the two closest indices
    idx = np.searchsorted(band_2_wavelength, wvl_nm)  # Returns insertion index

    if idx == 0:
        return 0  # Wavelength is at the first index
    elif idx == len(band_2_wavelength):
        return len(band_2_wavelength) - 1  # Wavelength is at the last index

    # Get the surrounding wavelength values
    wvl_low, wvl_high = band_2_wavelength[idx - 1], band_2_wavelength[idx]

    # Linear interpolation
    wvl_index = (idx - 1) + (wvl_nm - wvl_low) / (wvl_high - wvl_low)

    return int(round(wvl_index))
