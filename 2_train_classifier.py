from meridian_utils import train_classifier

main_folder = r'C:\Users\kzammit\Documents\Detector\20230913'
inputs_folder = main_folder + '\\' + 'inputs'
spec_name = 'spec_config.json'
db_name = r'database_20230913.h5'
checkpoint_folder = main_folder + '\\' + 'checkpoints'
log_folder = main_folder + '\\' + 'logs'
recipe_name = 'resnet_recipe.json'
classifier_name = 'rs-20230913.kt'


# create classifier .kt file containing the trained detector, with recipe and spectro files attached
train_classifier(spectro_file=inputs_folder + "\\" + spec_name,
                 database_h5=main_folder + "\\" + db_name,
                 recipe=inputs_folder + "\\" + recipe_name,
                 batch_size=16, n_epochs=20,
                 output_name=main_folder + "\\" + classifier_name,
                 checkpoint_folder=checkpoint_folder, log_folder=log_folder)
