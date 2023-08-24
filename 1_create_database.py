from meridian_utils import create_database, calc_file_durations

main_folder = r'C:\Users\kzammit\Documents\Detector\20230606'
inputs_folder = main_folder + '\\' + 'inputs'
data_folder = r'D:\ringed-seal-data'
annot_t_csv = "annotations_train.csv"
annot_v_csv = "annotations_val.csv"
spec_name = 'spec_config_karlee.json'
db_name = r'database_20230606.h5'
file_durations_file = r'C:\Users\kzammit\Documents\Detector\20230606\inputs\all_file_durations.xlsx'

calc_file_durations(data_folder)

'''
# create database.h5 file containing the test and validation files
create_database(train_csv=inputs_folder + '\\' + annot_t_csv,
                val_csv=inputs_folder + '\\' + annot_v_csv,
                spectro_file=inputs_folder + '\\' + spec_name,
                data_folder=data_folder,
                length=2.0,
                output_db_name=main_folder + '\\' + db_name, 
                file_durations_file = file_durations_file)

'''