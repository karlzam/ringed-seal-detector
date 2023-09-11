from meridian_utils import create_detector

main_folder = r'C:\Users\kzammit\Documents\Detector\20230906'
classifier_name = 'ringed_seal_test_20230907.kt'
audio_folder = r'D:\ringed-seal-data\Ulu_2022'
temp_folder_name = 'ringedS_tmp_folder'
detections_csv = main_folder + '\\' + 'detections.csv'
spec_folder = main_folder + '\\' + 'inputs' + '\\' + 'spec_config_noRate.json'

# use detector on audio data
create_detector(model_file=main_folder + '\\' + classifier_name,
                temp_model_folder=main_folder + '\\' + temp_folder_name,
                detections_csv=detections_csv,
                audio_folder=audio_folder,
                spec_folder=spec_folder,
                threshold=0.5, step_size=2.0, batch_size=64, buffer=0)
