"""
This script contains all the utilities created before training the detector. Methods included:
- Scatter plot of call lengths
- Plot spectrograms
- Write file locations from data  directory
"""
from toolbox_utils import plot_call_length_scatter, write_file_locations, rename_ulu_2022_files, inspect_audio_files

### Scatter Plot of Call Lengths ###
#annotations_table = r'C:\Users\kzammit\Documents\Detector\formatted_annot_20230518.xlsx'
#output_fig_folder = r'C:\Users\kzammit\Documents\Detector'
#plot_call_length_scatter(annotations_table, output_fig_folder, all_combined=0)

### Write File Locations from Data Dir ###
#input_folder = r'D:\ringed-seal-data'
#output_file_name = r'C:\Users\kzammit\Documents\Detector\original_file_locations.xlsx'
#write_file_locations(input_folder, output_file_name)

### Rename Ulu 2022 Files ###
#data_folder = r'D:\ringed-seal-data\Ulu_2022'
#annotation_tables = r'D:\ringed_seal_selection_tables\2023_06_13_All_completed_RS_selection_Tables_by_MB\Completed' \
#                    r'\odd_ducky'
#rename_ulu_2022_files(data_folder, annotation_tables)

### Inspect audio files ###

wav = r'D:\ringed-seal-data\Cape_Bathurst_50_2018_2019\1208795168.700209104318.wav'

sig, rate = inspect_audio_files(wav)
