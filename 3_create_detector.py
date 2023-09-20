from meridian_utils import create_detector
from toolbox_utils import copy_audio_files

main_folder = r'C:\Users\kzammit\Documents\Detector\20230913'
classifier_name = 'rs-20230920.kt'
audio_folder = main_folder + '\\' + 'audio'
test_csv = main_folder + '\\' + 'inputs' + '\\' + 'annotations_test.csv'
temp_folder_name = 'ringedS_tmp_folder'
detections_csv = main_folder + '\\' + 'detections_raw.csv'
spec_folder = main_folder + '\\' + 'inputs' + '\\' + 'spec_config.json'

# copy the validation audio files to the correct folder (can copy out if you've already done this step for this
# annotation file)
copy_audio_files(test_csv, audio_folder)

# use detector on audio data
#create_detector(model_file=main_folder + '\\' + classifier_name,
##                temp_model_folder=main_folder + '\\' + temp_folder_name,
#                detections_csv=detections_csv,
#                audio_folder=audio_folder,
#                spec_folder=spec_folder,
#                threshold=0.5, step_size=2.0, batch_size=16, buffer=0.5)



