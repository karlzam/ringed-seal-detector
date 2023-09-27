from model_testing_utils import compare, compute_model_outcome, make_confusion_matrix

main_folder = r'C:\Users\kzammit\Documents\Detector\20230913'
db_name = main_folder + '\\' + 'database_20230920.h5'
model_name = main_folder + '\\' + 'rs-20230920.kt'
new_model_folder = main_folder + '\\' + 'saved_model\RS_temp_folder'
output_dir = main_folder + '\\' + 'metrics'
test_annot_csv = main_folder + '\\' + 'inputs' + r'\annotations_test.csv'
detections = main_folder + '\\' + r'detections.csv'
classifications_file = output_dir + '\\' + 'classifications.csv'

comparison = compare(test_annot_csv, detections)

comparison.to_excel(main_folder + r'\detected_annotations.xlsx')

compute_model_outcome(model_name, db_name, new_model_folder, output_dir, batch_size=16, threshold=0.5,
                      table_name='/test')

make_confusion_matrix(classifications_file, output_dir)
