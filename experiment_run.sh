#!/bin/bash

python3 gref4hsi/tests/test_main_dbe.py --data_dir="/home/leo/Documents/NTNU/Masterthesis/Data/UHI_Data/Gref_processed/Svea1_Day1/Transect_6/Gref_10mm_DVL_minimal_inter/" --resolution=0.01 --interpolation=True --alti_data="alti" --raster_transform="minimal_rectangle";

#python3 gref4hsi/tests/test_main_dbe.py --data_dir="/home/leo/Documents/NTNU/Masterthesis/Data/UHI_Data/Gref_processed/Svea3_Day1/Gref_Transect_1c/Gref_10mm_DVL_minimal_inter/" --resolution=0.01 --interpolation=True --alti_data="dvl" --raster_transform="minimal_rectangle";

#python3 gref4hsi/tests/test_main_dbe.py --data_dir="/home/leo/Documents/NTNU/Masterthesis/Data/UHI_Data/Gref_processed/Svea3_Day1/Gref_Transect_1e/Gref_10mm_DVL_minimal_inter/" --resolution=0.01 --interpolation=True --alti_data="dvl" --raster_transform="minimal_rectangle";

python3 gref4hsi/tests/test_main_dbe.py --data_dir="/home/leo/Documents/NTNU/Masterthesis/Data/UHI_Data/Gref_processed/Svea3_Day1/Gref_Transect_2a/Gref_10mm_DVL_minimal_inter/" --resolution=0.01 --interpolation=True --alti_data="dvl" --raster_transform="minimal_rectangle";

#python3 gref4hsi/tests/test_main_dbe.py --data_dir="/home/leo/Documents/NTNU/Masterthesis/Data/UHI_Data/Gref_processed/Svea3_Day1/Gref_Transect_2d/Gref_10mm_DVL_minimal_inter/" --resolution=0.01 --interpolation=True --alti_data="dvl" --raster_transform="minimal_rectangle";

python3 gref4hsi/tests/test_main_dbe.py --data_dir="/home/leo/Documents/NTNU/Masterthesis/Data/UHI_Data/Gref_processed/Svea3_Day1/Gref_Transect_2e/Gref_10mm_DVL_minimal_inter/" --resolution=0.01 --interpolation=True --alti_data="dvl" --raster_transform="minimal_rectangle";



# Liu testing
# svea3 (missing alt. data)
python gref4hsi/tests/test_main_dbe.py --data_dir="C:\Users\Erik Liu\OneDrive - NTNU\PhD\Courses\UHI post processing\havard_repo\from_Leo_usb\Svea3_Day1\Gref_Transect_2a\Gref_10mm_DVL_minimal_inter" --resolution=0.01 --interpolation=True --alti_data="dvl" --raster_transform="minimal_rectangle";

# svea2 (dont have hsi data)
python gref4hsi/tests/test_main_dbe.py --data_dir="C:\Users\Erik Liu\OneDrive - NTNU\PhD\Courses\UHI post processing\havard_repo\from_Leo_usb\Fieldtest_Svea_2\Gref_10mm_alti_north_east_inter" --resolution=0.01 --interpolation=True --alti_data="dvl" --raster_transform="minimal_rectangle";

# svea2_Day1 
python gref4hsi/tests/test_main_dbe.py --data_dir="E:\mjosa\Svea2_Day_1\Gref_Transect_2\Gref_10mm_DVL_minimal_inter" --resolution=0.01 --interpolation=True --alti_data="dvl" --raster_transform="minimal_rectangle";
# decrease res to make run faster
#! THIS WORKED LAST TIME EXCEPT THE ORTHORECTIFICATION
python gref4hsi/tests/test_main_dbe.py --data_dir="E:\mjosa\Svea2_Day_1\Gref_Transect_2\Gref_10mm_DVL_minimal_inter" --resolution=10 --interpolation=True --alti_data="dvl" --raster_transform="minimal_rectangle"



# python gref4hsi/tests/test_eely.py \
#   --data_dir="E:\mjosa\Svea2_Day_1\Gref_Transect_2\Gref_10mm_DVL_minimal_inter" \
#   --csv_nav_path="E:\mjosa_new\navigation_data\105049\nav_data_105049_dr.csv" \
#   --resolution=0.01 \
#   --interpolation=True \
#   --raster_transform="minimal_rectangle" \
#   --skip_raw_processing \
#   --create_dem_from_csv \
#   --dem_resolution=0.2


python gref4hsi/tests/test_eely.py --data_dir="E:\mjosa_new\use_gref4hsi\221" --csv_nav_path="E:\mjosa_new\navigation_data\105049\nav_data_105049_dr.csv" --resolution=0.01 --interpolation=True --raster_transform="minimal_rectangle" --time_offset_sec=510