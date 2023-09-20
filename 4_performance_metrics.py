from meridian_utils import compare, compute_model_outcome, make_confusion_matrix

main_folder = r'C:\Users\kzammit\Documents\Detector\20230913'
db_name = main_folder + '\\' + 'database_20230920.h5'
model_name = main_folder + '\\' + 'rs-20230920.kt'
new_model_folder = main_folder + '\\' + 'saved_model\RS_temp_folder'
output_dir = main_folder + '\\' + 'ruwan_stuff'
metrics_file = r'C:\Users\kzammit\Documents\Detector\20230913\ruwan_stuff\metrics.csv'

#comparison = compare(test_annot_csv, detections)

#comparison.to_excel(main_folder + r'\detected_annotations.xlsx')

#compute_model_outcome(model_name, db_name, new_model_folder, output_dir, batch_size=16, threshold=0.5)

make_confusion_matrix(metrics_file)