"""
This script contains all the utilities created before training the detector. Methods included:
- Scatter plot of call lengths
- Plot spectrograms
- Write file locations from data  directory
"""
from toolbox_utils import plot_call_length_scatter, plot_spectrograms, write_file_locations

### Scatter Plot of Call Lengths ###
#annotations_table = r'C:\Users\kzammit\Documents\Detector\formatted_annot_20230518.xlsx'
#output_fig_folder = r'C:\Users\kzammit\Documents\Detector'
#plot_call_length_scatter(annotations_table, output_fig_folder, all_combined=0)

### Plot Spectrograms ###
annot_file = r'C:\Users\kzammit\Repos\ringed-seal-meridian\train_selections_20230529.xlsx'
data_dir = r"D:\ringed-seal-data"
output_dir = r''
plot_spectrograms(annot_file, data_dir, output_dir, plot_examples=1, desired_label=1)

### Write File Locations from Data Dir ###
#input_folder = r'D:\ringed-seal-data'
#output_file_name = r'C:\Users\kzammit\Documents\Detector\original_file_locations.xlsx'
#write_file_locations(input_folder, output_file_name)