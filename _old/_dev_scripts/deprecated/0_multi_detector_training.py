from meridian_utils import create_database, train_classifier, create_detector
from toolbox_utils import copy_audio_files
from model_testing_utils import compare, compute_model_outcome, make_confusion_matrix

# All
num_tests = 2
main_folder = r'C:\Users\kzammit\Documents\Detector\spec-dur-tests'
inputs_folder = r'C:\Users\kzammit\Documents\Detector\spec-dur-tests\inputs'
data_folder = r'D:\ringed-seal-data'
annot_tr_csv = "annotations_train.csv"
annot_v_csv = "annotations_val.csv"
annot_te_csv = "annotations_test.csv"
file_durations_file = r'C:\Users\kzammit\Documents\Detector\20230913\inputs\all_file_durations_complete.xlsx'
recipe = inputs_folder + "\\" + 'resnet_recipe.json'
temp_folder_name = 'ringedS_tmp_folder'
audio_folder = main_folder + '\\' + 'audio'
test_csv = inputs_folder + '\\' + 'annotations_test.csv'
test_annot_csv = main_folder + '\\' + 'inputs' + r'\annotations_test.csv'

# Run specific
sub_folders = [main_folder + '\\' + '1sec', main_folder + '\\' + '3sec']
spec_name = ['spec_config_1sec.json', 'spec_config_3sec.json']
db_name = [r'database_1sec.h5', r'database_3sec.h5']
lengths = [1.0, 3.0]
classifier_names = ['rs-1sec.kt', 'rs-3sec.kt']


# copy the validation audio files to the correct folder (can copy out if you've already done this step for this
# annotation file)
#copy_audio_files(test_csv, audio_folder)

for idx, folder in enumerate(sub_folders):

    checkpoint_folder = folder + '\\' + 'checkpoints'
    log_folder = folder + '\\' + 'logs'
    new_model_folder = folder + '\\' + 'saved_model\RS_temp_folder'
    output_dir = folder + '\\' + 'metrics'
    classifications_file = output_dir + '\\' + 'classifications.csv'
    db_name_x = str(folder) + '\\' + db_name[idx]
    spectro_file = inputs_folder + '\\' + spec_name[idx]
    model_name = str(folder) + "\\" + classifier_names[idx]
    temp_folder = folder + '\\' + temp_folder_name
    detections = folder + '\\' + r'detections_raw.csv'

    '''

    create_database(train_csv=inputs_folder + '\\' + annot_tr_csv,
                    val_csv=inputs_folder + '\\' + annot_v_csv,
                    test_csv=inputs_folder + '\\' + annot_te_csv,
                    spectro_file=spectro_file,
                    data_folder=data_folder,
                    length=lengths[idx],
                    output_db_name=db_name_x,
                    file_durations_file=file_durations_file)

    train_classifier(spectro_file=spectro_file,
                     database_h5=db_name_x,
                     recipe=recipe,
                     batch_size=16, n_epochs=20,
                     output_name=model_name,
                     checkpoint_folder=checkpoint_folder, log_folder=log_folder)

    create_detector(model_file=model_name,
                    temp_model_folder=temp_folder,
                    detections_csv=folder + '\\' + 'detections_raw.csv',
                    audio_folder=audio_folder,
                    spec_folder=spectro_file,
                    threshold=0.5, step_size=2.0, batch_size=16, buffer=0.5)
    '''

    comparison = compare(test_annot_csv, detections)
    comparison.to_excel(folder + r'\detected_annotations.xlsx')
    compute_model_outcome(model_name, db_name_x,
                          new_model_folder, output_dir, batch_size=16, threshold=0.5,
                        table_name='/test')
    make_confusion_matrix(classifications_file, output_dir)


