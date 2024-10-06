import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import shutil
import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as confusion_matrix_sklearn
import glob

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

print('done importing packages')

# Pathway to folder containing formatted complete annotation tables by site
formatted_annot_folder = r'C:\Users\kzammit\Documents\Detector\manual_dataset_build\formatted_annots'

# Get list of all csv files in that folder
files = glob.glob(formatted_annot_folder + "/*.csv")

# Set length of spectrogram to be 1 sec
length = 1.0

output_path = r'C:\Users\kzammit\Documents\Detector\detector-1sec-timeShift\annots'

# For each csv file
for file in files:

    annots = pd.read_csv(file)
    std_annot = sl.standardize(table=annots, labels=["B", "BY"], start_labels_at_1=True, trim_table=True)
    std_annot['label'] = std_annot['label'].replace(2, 1)
    print('Training data standardized? ' + str(sl.is_standardized(std_annot)))

    # Step: Produce multiple selections for each annotated section by shifting the selection
    # window in steps of length step (in seconds) both forward and backward in
    # time. The default value is 0.

    # Min Overlap: Minimum required overlap between the selection and the annotated section, expressed
    # as a fraction of whichever of the two is shorter. Only used if step > 0.

    positives = sl.select(annotations=std_annot, length=length, step=0.3, min_overlap=1, center=False)

    new_file_name = output_path + '\\' + file.split('\\')[-1].split('.')[0] + '_shifted.csv'

    positives.to_csv(new_file_name)



















