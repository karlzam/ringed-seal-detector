"""
This script contains spectrogram investigation functions:
- Plot spectrograms

"""
from spectrogram_utils import load_audio_seg, plot_spectrogram

### Plot Spectrograms ###
annot_file = r'C:\Users\kzammit\Documents\Detector\20230906\inputs\all_annotations_20230830.xlsx'
data_dir = r"D:\ringed-seal-data"
output_dir = r'D:\spectrograms\Rate'
spec_file = r'C:\Users\kzammit\Documents\Detector\20230906\inputs\spec_config_karlee_noUnits.json'

# create an audioloader with the spectrograms
# note you need to define step, maximum overlap, and length in this step
annot, loader = load_audio_seg(annot_file, spec_file, data_dir)

# take audio segments and create spectrogram representation
plot_spectrogram(annot, loader, output_dir)
