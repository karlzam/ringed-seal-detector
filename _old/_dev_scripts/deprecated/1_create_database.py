from meridian_utils import create_database
from toolbox_utils import calc_file_durations

main_folder = r'C:\Users\kzammit\Documents\Detector\20230913'
inputs_folder = main_folder + '\\' + 'inputs'
data_folder = r'D:\ringed-seal-data'
annot_tr_csv = "annotations_train.csv"
annot_v_csv = "annotations_val.csv"
annot_te_csv = "annotations_test.csv"
spec_name = 'spec_config.json'
db_name = r'database_20230920.h5'
file_durations_file = r'C:\Users\kzammit\Documents\Detector\20230913\inputs\all_file_durations_complete.xlsx'

#calc_file_durations(data_folder)

# create database.h5 file containing the test and validation filesx
create_database(train_csv=inputs_folder + '\\' + annot_tr_csv,
                val_csv=inputs_folder + '\\' + annot_v_csv,
                test_csv = inputs_folder + '\\' + annot_te_csv,
                spectro_file=inputs_folder + '\\' + spec_name,
                data_folder=data_folder,
                length=2.0,
                output_db_name=main_folder + '\\' + db_name, 
                file_durations_file=file_durations_file)
