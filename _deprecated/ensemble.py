import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import shutil
import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as confusion_matrix_sklearn

from ketos.data_handling import selection_table as sl
import ketos.data_handling.database_interface as dbi
from ketos.data_handling.parsing import load_audio_representation
from ketos.data_handling.data_feeding import BatchGenerator
from ketos.neural_networks.resnet import ResNetInterface
from ketos.audio.audio_loader import AudioFrameLoader, AudioLoader, SelectionTableIterator
from ketos.audio.spectrogram import MagSpectrogram
from ketos.neural_networks.dev_utils.detection import batch_load_audio_file_data, filter_by_threshold
from ketos.data_handling.data_feeding import JointBatchGen

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

main_folder = r'C:\Users\kzammit\Documents\Detector\detector-1sec'
temp_folder_base = main_folder + '\\' + 'ringedS_tmp_folder-'
scores_csv_base = main_folder + '\\' + 'scores_raw-'
model_name_base = main_folder + '\\' + 'rs-1sec-'
num_models = 5
spec_file = main_folder + '\\' + r'inputs\spec_config_1sec.json'
audio_folder = main_folder + '\\' + 'audio'
db_name = main_folder + '\\' + 'manual_database_1sec.h5'

# Open the database in read only file
db = dbi.open_file(db_name, 'r')

# Open the table in the database at the root level
table = dbi.open_table(db, '/test')

all_df = pd.DataFrame()

batch_size = 16

for idx in range(0, num_models):

    model_name = model_name_base + str(idx+1) + '.kt'
    scores_csv = scores_csv_base + str(idx+1) + '.csv'
    temp_folder = temp_folder_base + str(idx+1)

    model = ResNetInterface.load(model_file=model_name, load_audio_repr=False, new_model_folder=temp_folder)

    # Create a batch generator
    gens = []

    # Calculate the batch_size fixing the original batch size so there are no remainders
    batch_size = int(batch_size / sum(1 for _ in db.walk_nodes(table, "Table")))

    # for the testing dataset table in the database (or whatever table is passed in)
    for group in db.walk_nodes(table, "Table"):
        # Create a batch generator for this table
        generator = BatchGenerator(batch_size=batch_size, data_table=group,
                                   output_transform_func=ResNetInterface.transform_batch, shuffle=False,
                                   refresh_on_epoch_end=False, x_field='data', return_batch_ids=True)

        # Append the generator to the gens array
        gens.append(generator)

    # Create a joint batch generator if multiple tables are passed through
    gen = JointBatchGen(gens, n_batches='min', shuffle_batch=False, reset_generators=False, return_batch_ids=True)

    # Initialize the scores and labels
    scores = []
    labels = []

    # For each batch in the joint batch generator
    for batch_id in range(gen.n_batches):
        # Get the ids, spectrograms, and labels for the data in the batch
        hdf5_ids, batch_X, batch_Y = next(gen)

        # Get the labels for the batch data, using the "argmax" func which returns the col header, so 0 is a noise
        # segment, 1 is a rs segment
        batch_labels = np.argmax(batch_Y, axis=1)

        # Returns the scores for the batch for the "positive" class - this is used in the compute detections
        # function later on
        batch_scores = model.model.predict_on_batch(batch_X)

        # Add these scores for this batch to the overall list
        scores.extend(batch_scores)
        labels.extend(batch_labels)

        print('test')

    # Create a numpy array for the labels and scores for all batches
    labels = np.array(labels)
    scores = np.array(scores)

    print('test')



