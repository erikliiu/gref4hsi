#!/bin/bash

python3 gref4hsi/tests/test_main_dbe.py --data_dir="/home/leo/Documents/NTNU/Masterthesis/Data/UHI_Data/Gref_processed/Svea1_Day1/Transect_6/Gref_10mm_DVL_minimal_inter/" --resolution=0.01 --interpolation=True --alti_data="alti" --raster_transform="minimal_rectangle";

#python3 gref4hsi/tests/test_main_dbe.py --data_dir="/home/leo/Documents/NTNU/Masterthesis/Data/UHI_Data/Gref_processed/Svea3_Day1/Gref_Transect_1c/Gref_10mm_DVL_minimal_inter/" --resolution=0.01 --interpolation=True --alti_data="dvl" --raster_transform="minimal_rectangle";

#python3 gref4hsi/tests/test_main_dbe.py --data_dir="/home/leo/Documents/NTNU/Masterthesis/Data/UHI_Data/Gref_processed/Svea3_Day1/Gref_Transect_1e/Gref_10mm_DVL_minimal_inter/" --resolution=0.01 --interpolation=True --alti_data="dvl" --raster_transform="minimal_rectangle";

python3 gref4hsi/tests/test_main_dbe.py --data_dir="/home/leo/Documents/NTNU/Masterthesis/Data/UHI_Data/Gref_processed/Svea3_Day1/Gref_Transect_2a/Gref_10mm_DVL_minimal_inter/" --resolution=0.01 --interpolation=True --alti_data="dvl" --raster_transform="minimal_rectangle";

#python3 gref4hsi/tests/test_main_dbe.py --data_dir="/home/leo/Documents/NTNU/Masterthesis/Data/UHI_Data/Gref_processed/Svea3_Day1/Gref_Transect_2d/Gref_10mm_DVL_minimal_inter/" --resolution=0.01 --interpolation=True --alti_data="dvl" --raster_transform="minimal_rectangle";

python3 gref4hsi/tests/test_main_dbe.py --data_dir="/home/leo/Documents/NTNU/Masterthesis/Data/UHI_Data/Gref_processed/Svea3_Day1/Gref_Transect_2e/Gref_10mm_DVL_minimal_inter/" --resolution=0.01 --interpolation=True --alti_data="dvl" --raster_transform="minimal_rectangle";



# Liu testing
python gref4hsi/tests/test_main_dbe.py --data_dir="C:\Users\Erik Liu\OneDrive - NTNU\PhD\Courses\UHI post processing\havard_repo\from_Leo_usb\Svea3_Day1\Gref_Transect_2a\Gref_10mm_DVL_minimal_inter" --resolution=0.01 --interpolation=True --alti_data="dvl" --raster_transform="minimal_rectangle";
 