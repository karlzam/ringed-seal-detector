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
import scipy

from ketos.data_handling import selection_table as sl
import ketos.data_handling.database_interface as dbi
from ketos.data_handling.parsing import load_audio_representation
from ketos.data_handling.data_feeding import BatchGenerator
from ketos.neural_networks.resnet import ResNetInterface
from ketos.audio.audio_loader import AudioFrameLoader, AudioLoader, SelectionTableIterator
from ketos.audio.spectrogram import MagSpectrogram
from ketos.neural_networks.dev_utils.detection import batch_load_audio_file_data, filter_by_threshold, filter_by_label, merge_overlapping_detections
from ketos.data_handling.data_feeding import JointBatchGen

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_batch_generator(spectro_file, audio_folder, step_size, batch_size):
    audio_repr = load_audio_representation(path=spectro_file)

    spec_config = audio_repr['spectrogram']

    audio_loader = AudioFrameLoader(path=audio_folder, duration=spec_config['duration'],
                                    step=step_size, stop=False, representation=spec_config['type'],
                                    representation_params=spec_config, pad=False)

    batch_generator = batch_load_audio_file_data(loader=audio_loader, batch_size=batch_size)

    return batch_generator


def load_models(model_names, temp_folders):
    models = []
    for idx, modelz in enumerate(model_names):
        models.append(ResNetInterface.load(model_file=modelz, new_model_folder=temp_folders[idx]))

    return models


def get_detections(batch_generator, models, output_dir, threshold, raven_txt, audio_folder):
    detections_pos = pd.DataFrame()
    detections_neg = pd.DataFrame()

    for ibx, batch_data in enumerate(batch_generator):

        for idx, model in enumerate(models):

            # Run the model on the spectrogram data from the current batch
            batch_predictions = model.run_on_batch(batch_data['data'], return_raw_output=True)

            if idx == 0:
                # Lets store our data in a dictionary

                raw_output_neg = {'filename': batch_data['filename'], 'start': batch_data['start'],
                                  'end': batch_data['end'], '0-0': batch_predictions[:, 0]}

                raw_output_pos = {'filename': batch_data['filename'], 'start': batch_data['start'],
                                  'end': batch_data['end'], '1-0': batch_predictions[:, 1]}

            else:
                raw_output_neg |= {'0-' + str(idx): batch_predictions[:, 0]}

                raw_output_pos |= {'1-' + str(idx): batch_predictions[:, 1]}

        detections_pos = pd.concat([detections_pos, pd.DataFrame.from_dict(raw_output_pos)])
        detections_neg = pd.concat([detections_neg, pd.DataFrame.from_dict(raw_output_neg)])

        print('test')

    detections_pos.to_excel(output_dir + '\\' + 'detections-pos.xlsx', index=False)
    detections_neg.to_excel(output_dir + '\\' + 'detections-neg.xlsx', index=False)

    mean_cols_pos = detections_pos.columns[3:]
    mean_cols_neg = detections_neg.columns[3:]

    detections_pos['mean-pos'] = detections_pos[mean_cols_pos].mean(axis=1)
    detections_neg['mean-neg'] = detections_neg[mean_cols_neg].mean(axis=1)

    merge_df = detections_pos[['filename', 'start', 'end', 'mean-pos']].copy()
    merge_df['mean-neg'] = detections_neg['mean-neg']

    scores = []
    for row in merge_df.iterrows():
        score = [row[1]['mean-neg'], row[1]['mean-pos']]
        scores.extend([score])

    dict = {'filename': merge_df['filename'], 'start': merge_df['start'], 'end': merge_df['end'], 'score': scores}

    filter_detections = filter_by_threshold(dict, threshold=threshold)
    detections_filtered = filter_by_label(filter_detections, labels=1).reset_index(drop=True)
    print(len(detections_filtered))
    detections_grp = merge_overlapping_detections(detections_filtered)
    print(len(detections_grp))

    results_table = detections_grp

    cols = ['filename']
    results_table.loc[:, cols] = results_table.loc[:, cols].ffill()
    results_table['Selection'] = results_table.index + 1
    results_table['View'] = 'Spectrogram 1'
    results_table['Channel'] = 1
    results_table['Begin Path'] = audio_folder + '\\' + results_table.filename
    results_table['File Offset (s)'] = results_table.start
    results_table = results_table.rename(
        columns={"start": "Begin Time (s)", "end": "End Time (s)", "filename": "Begin File"})
    results_table['Begin File'] = results_table['Begin File']
    results_table['Low Freq (Hz)'] = 100
    results_table['High Freq (Hz)'] = 1200

    results_table.to_csv(raven_txt, index=False, sep='\t')

    return detections_grp

main_folder = r'E:\baseline-with-normalization-reduce-tonal\ulu2023\detections\ensemble'

model_folder = r'E:\baseline-with-normalization-reduce-tonal\models'

model_names = [model_folder + "\\" + "rs-model-0.kt", model_folder + "\\" + "rs-model-1.kt", model_folder + "\\" + "rs-model-2.kt",
            model_folder + "\\" + "rs-model-3.kt", model_folder + "\\" + "rs-model-4.kt", model_folder + "\\" + "rs-model-5.kt",
            model_folder + "\\" + "rs-model-6.kt", model_folder + "\\" + "rs-model-7.kt", model_folder + "\\" + "rs-model-8.kt",
            model_folder + "\\" + "rs-model-9.kt"]

temp_folders = [model_folder + "\\" + "temp-0", model_folder + "\\" + "temp-1", model_folder + "\\" + "temp-2",
            model_folder + "\\" + "temp-3", model_folder + "\\" + "temp-4", model_folder + "\\" + "temp-5",
            model_folder + "\\" + "temp-6", model_folder + "\\" + "temp-7", model_folder + "\\" + "temp-8",
            model_folder + "\\" + "temp-9"]

spectro_file = r'E:\baseline-with-normalization-reduce-tonal\spec_config_100-1200Hz-0.032-hamm-normalized-reduce-tonal.json'
output_dir = main_folder
audio_folder = r'D:\ringed-seal-data\Ulu_2023_St5_Site65\test-subset'
detections_csv = output_dir + '\\' + 'detections-avg.csv'
temp_folder = output_dir + '\\' + 'ringedS_tmp_folder'
pos_detection = output_dir + '\\' + 'grouped-filtered-dets.xlsx'
raven_txt = output_dir + '\\' + 'raven-formatted-detections.txt'

# Step 0.5s each time (overlap of 50% for 1 sec duration)
step_size = 0.5

# Number of samples in batch
batch_size = 16

# Threshold
threshold = 0.5

batch_generator = get_batch_generator(spectro_file, audio_folder, step_size, batch_size)
all_models = load_models(model_names, temp_folders)
detections_grp = get_detections(batch_generator, all_models, output_dir, threshold, raven_txt, audio_folder)