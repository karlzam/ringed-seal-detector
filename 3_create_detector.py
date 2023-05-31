from meridian_utils import create_detector

main_folder = r'C:\Users\kzammit\Documents\Detector\20230531'
classifier_name = 'ringed_seal_test_20230531.kt'
audio_folder = main_folder + '\\' + r'audio'
temp_folder_name = 'ringedS_tmp_folder'
detections_csv = main_folder + '\\' + 'detections.csv'

# use detector on audio data
create_detector(model_file=main_folder + '\\' + classifier_name,
                temp_model_folder=main_folder + '\\' + temp_folder_name,
                detections_csv=detections_csv,
                audio_folder=audio_folder,
                threshold=0.4, step_size=0.5, batch_size=16, buffer=1.5)
