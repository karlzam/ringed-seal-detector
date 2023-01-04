from meridian_utils import create_database, train_classifier, create_detector

main_folder = r'C:\Users\kzammit\Documents\Work\test'

# database folders and names
input_file_folder_db = main_folder + '\\' + 'database\input'
output_file_folder_db = main_folder + '\\' + 'database\output'
data_folder = main_folder + '\\' + 'database\input\data'
annott_csv_name = "annotations_train.csv"
annotv_csv_name = "annotations_val.csv"
spec_name = 'spec_config.json'
db_name = r'database_RS.h5'

# classifier folders and names
input_file_folder_c = main_folder + '\\' + 'classifier\input'
output_file_folder_c = main_folder + '\\' + 'classifier\output'
checkpoint_folder = main_folder + '\\' + 'classifier\output\checkpoints'
recipe_name = 'recipe.json'
classifier_name = 'ringed_seal_test.kt'


# detector folders and names
input_file_folder_d = main_folder + '\\' + 'detector\input'
audio_folder_d = main_folder + '\\' + r'detector\audio'
output_file_folder_d = main_folder + '\\' + 'detector\output'
temp_folder_name = 'ringedS_tmp_folder'
detections_csv = main_folder + '\\' + 'detector\output\detections.csv'

'''
# create database.h5 file containing the test and validation files
create_database(train_csv=input_file_folder_db + '\\' + annott_csv_name,
                val_csv=input_file_folder_db + '\\' + annotv_csv_name,
                spectro_file=input_file_folder_db + '\\' + spec_name,
                data_folder=data_folder,
                length=3.0,
                output_db_name=output_file_folder_db + '\\' + db_name)
'''

'''
# create classifier .kt file containing the trained detector, with recipe and spectro files attached
train_classifier(spectro_file=input_file_folder_db + r'\\' + spec_name,
                 database_h5=output_file_folder_db + r'\\' + db_name,
                 recipe=input_file_folder_c + r'\\' + recipe_name,
                 batch_size=48, n_epochs=5,
                 output_name=output_file_folder_c + "\\" + classifier_name,
                 checkpoint_folder=checkpoint_folder)

'''


# use detector on audio data
create_detector(model_file=output_file_folder_c + '\\' + classifier_name,
                temp_model_folder=output_file_folder_d + '\\' + temp_folder_name,
                detections_csv=detections_csv,
                audio_folder=audio_folder_d,
                threshold=0.70, step_size=3.0, batch_size=64, buffer=1.5)
