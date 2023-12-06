import numpy as np
import tensorflow as tf
import ketos.data_handling.database_interface as dbi
from ketos.data_handling.data_feeding import BatchGenerator
from ketos.neural_networks.resnet import ResNetInterface

main_folder = r'C:\Users\kzammit\Documents\Detector\manual_detector_2sec'
db_name = main_folder + '\\' r'manual_database_2sec.h5'
recipe = main_folder + '\\' + r'inputs\resnet_recipe.json'
output_name = main_folder + '\\' + 'rs-2sec-test.kt'
spec_file = main_folder + '\\' + r'inputs\spec_config_2sec.json'


## Train Classifier ##
# Set the random seeds for numpy and tensorflow for reproducibility
np.random.seed(1000)
tf.random.set_seed(2000)

# Set the batch size and number of epochs for training
batch_size = 16
n_epochs = 5
log_folder = main_folder + '\\' + 'logs'
checkpoint_folder = main_folder + '\\' + 'checkpoints'

# Open the database
db = dbi.open_file(db_name, 'r')

# Open the training and validation tables from the database
train_data = dbi.open_table(db, "/train/data")
val_data = dbi.open_table(db, "/val/data")

# Create training batches
train_generator = BatchGenerator(batch_size=batch_size, data_table=train_data,
                                 output_transform_func=ResNetInterface.transform_batch,
                                 shuffle=True, refresh_on_epoch_end=True)

val_generator = BatchGenerator(batch_size=batch_size, data_table=val_data,
                               output_transform_func=ResNetInterface.transform_batch,
                               shuffle=False, refresh_on_epoch_end=False)

resnet = ResNetInterface.build(recipe)
resnet.train_generator = train_generator
resnet.val_generator = val_generator
resnet.log_dir = log_folder
resnet.checkpoint_dir = checkpoint_folder
resnet.train_loop(n_epochs=n_epochs, verbose=True, log_csv=True, csv_name='log.csv')
db.close()
resnet.save(output_name, audio_repr_file=spec_file)