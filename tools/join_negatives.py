import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import shutil
import os
import glob
import csv
import json
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

print('done importing packages')


folder = r"C:\Users\kzammit\Documents\Detector\detector-1sec\inputs\annots\negs-new\prev_manual_dataset_annots"

files = glob.glob(folder + "/*.xlsx")

new_negs = r'C:\Users\kzammit\Documents\Detector\detector-1sec\inputs\annots\negs-new\edits'

join_files = glob.glob(new_negs + '/*.csv')

output_dir = r'C:\Users\kzammit\Documents\Detector\detector-1sec\inputs\annots\negs-new\join'

names = ['CB', 'KK', 'PP', 'ULU2022', 'ULU']

for idex, file in enumerate(files):

    annots = pd.read_excel(file)
    df_annot = pd.DataFrame(columns=annots.columns)
    annots = annots.ffill()
    annots['end'] = annots['start'] + 1

    print('test')
    join_file = pd.read_csv(join_files[idex])
    annots = annots.rename(columns={"annot_id": "sel_id"})
    joined = pd.concat([join_file, annots])

    # See if there were duplicates bc generated separately
    joined['dup'] = joined.duplicated(subset=['filename', 'sel_id'])
    joined = joined.sort_values(by=['filename'])

    # Fix the selection id's for added ones... just gunna redo them, durg
    # get the names of each unique wav file
    unique_files = joined['filename'].unique()

    # loop through each unique wav, and set the annot_id for each one
    for fdex, wavF in enumerate(unique_files):

        # create a temp df that only has the entries for this wav file
        df_temp = joined[joined.filename == wavF]

        # reset the index
        df_temp = df_temp.reset_index(drop=True)

        # set the annot_id column to 'not set' initially
        df_temp['annot_id'] = 'not set'

        # start the counter at 0
        annot_id = 0

        # for the number of annotations with this wav file,
        for ii in range(0, len(df_temp)):
            # set the annot_id incrementally
            df_temp['annot_id'][ii] = annot_id
            annot_id += 1

        df_annot = pd.concat([df_annot, df_temp], ignore_index=True)

    df_annot = df_annot.drop(columns={'sel_id'})
    df_annot = df_annot.rename(columns={"annot_id": "sel_id"})

    df_annot.to_excel(output_dir + '\\' + names[idex] + '-negs-joined.xlsx', index=False)


