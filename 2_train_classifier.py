from meridian_utils import train_classifier

main_folder = r'C:\Users\kzammit\Documents\Detector\20230531'
inputs_folder = main_folder + '\\' + 'inputs'
spec_name = 'spec_config_karlee.json'
db_name = r'database_20230531.h5'
checkpoint_folder = main_folder + '\\' + 'checkpoints'
recipe_name = 'recipe.json'
classifier_name = 'ringed_seal_test_20230531.kt'

# create classifier .kt file containing the trained detector, with recipe and spectro files attached
train_classifier(spectro_file=inputs_folder + r'\\' + spec_name,
                 database_h5=main_folder + r'\\' + db_name,
                 recipe=inputs_folder + r'\\' + recipe_name,
                 batch_size=16, n_epochs=10,
                 output_name=main_folder + "\\" + classifier_name,
                 checkpoint_folder=checkpoint_folder)
