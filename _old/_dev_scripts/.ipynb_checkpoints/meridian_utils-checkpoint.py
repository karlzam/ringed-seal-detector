import ketos.audio
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ketos.data_handling import selection_table as sl
import ketos.data_handling.database_interface as dbi
from ketos.data_handling.parsing import load_audio_representation
from ketos.data_handling.data_feeding import BatchGenerator, JointBatchGen
from ketos.neural_networks.resnet import ResNetInterface
from ketos.audio.spectrogram import MagSpectrogram
from ketos.audio.audio_loader import AudioFrameLoader, AudioLoader, SelectionTableIterator
# from ketos.neural_networks.dev_utils.detection import process, save_detections
# from ketos.neural_networks.dev_utils.detection import save_detections
from ketos.neural_networks.dev_utils.detection import batch_load_audio_file_data, add_detection_buffer, compute_score_running_avg, merge_overlapping_detections, filter_by_threshold, filter_by_label


def create_database(train_csv, val_csv, test_csv, length, output_db_name, spectro_file, data_folder, file_durations_file):
    """
    Create a database of spectrograms for the training and validation datasets from annotation table .csv files
    :param train_csv: annotation table for the training data in .csv format (; delimited)
    :param val_csv: annotation table for the validation data in .csv format (; delimited)
    :param length: length of segments selected from spectrogram representing the time needed to encompass most calls
    :param output_db_name: name of the database.h5 file, must include .h5, example, "ringed_seal_db.h5"
    :param spectro_file: .json file containing spectrogram information, example, "spec_config.json"
    """

    # read in the training and validation annotation csv files
    annot_train = pd.read_csv(train_csv)
    annot_val = pd.read_csv(val_csv)
    annot_test = pd.read_csv(test_csv)

    # standardize the training and validation annotation csv files to ketos format
    std_annot_train = sl.standardize(table=annot_train, labels=["B", "BY"], start_labels_at_1=True, trim_table=True)
    # force labels all to be 1 for binary classification
    std_annot_train['label'] = std_annot_train['label'].replace(2, 1)

    std_annot_val = sl.standardize(table=annot_val, labels=["B", "BY"], start_labels_at_1=True, trim_table=True)
    std_annot_val['label'] = std_annot_val['label'].replace(2, 1)

    std_annot_test = sl.standardize(table=annot_test, labels=["B", "BY"], start_labels_at_1=True, trim_table=True)
    std_annot_test['label'] = std_annot_test['label'].replace(2, 1)

    print('Remember you are forcing all labels to be 1!')
    print('Training data standardized? ' + str(sl.is_standardized(std_annot_train)))
    print('Validation data standardized? ' + str(sl.is_standardized(std_annot_val)))
    print('Testing data standardized? ' + str(sl.is_standardized(std_annot_test)))


    # create segments of uniform length from the annotations tables
    positives_train = sl.select(annotations=std_annot_train, length=length, step=0.5, min_overlap=0.8, center=False)
    positives_val = sl.select(annotations=std_annot_val, length=length, step=0.0, center=False)
    positives_test = sl.select(annotations=std_annot_test, length=length, step=0.0, center=False)

    # read in the file durations file
    file_durations = pd.read_excel(file_durations_file)

    # drop rows in file durations that do not correspond to those wav files
    file_durations_train = file_durations[file_durations['filename'].isin(annot_train['filename'])]
    file_durations_val = file_durations[file_durations['filename'].isin(annot_val['filename'])]
    file_durations_test = file_durations[file_durations['filename'].isin(annot_test['filename'])]

    # generate negative segments (the same number as the positive segments),
    negatives_train = sl.create_rndm_selections(annotations=std_annot_train, files=file_durations_train,
                                                length=length, num=len(positives_train), trim_table=True)

    negatives_val = sl.create_rndm_selections(annotations=std_annot_val, files=file_durations_val,
                                              length=length, num=len(positives_val), trim_table=True)

    negatives_test = sl.create_rndm_selections(annotations=std_annot_test, files=file_durations_test,
                                              length=length, num=len(positives_test), trim_table=True)

    # drop selections that go past the end of the file
    positives_train = drop_out_of_bounds_sel(positives_train, file_durations_train)
    positives_val = drop_out_of_bounds_sel(positives_val, file_durations_val)
    positives_test = drop_out_of_bounds_sel(positives_test, file_durations_test)

    negatives_train = drop_out_of_bounds_sel(negatives_train, file_durations_train)
    negatives_val = drop_out_of_bounds_sel(negatives_val, file_durations_val)
    negatives_test = drop_out_of_bounds_sel(negatives_test, file_durations_test)

    # join the positive and negative vals together
    selections_train = pd.concat([positives_train, negatives_train], sort=False)
    selections_train.to_excel(r'C:\Users\kzammit\Documents\Detector\20230913\inputs\train_selections_20230920.xlsx')

    selections_val = pd.concat([positives_val, negatives_val], sort=False)
    selections_val.to_excel(r'C:\Users\kzammit\Documents\Detector\20230913\inputs\val_selections_20230920.xlsx')

    selections_test = pd.concat([positives_test, negatives_test], sort=False)
    selections_test.to_excel(r'C:\Users\kzammit\Documents\Detector\20230913\inputs\test_selections_20230920.xlsx')

    # load in the spectrogram settings
    spec_cfg = load_audio_representation(spectro_file, name="spectrogram")

    # compute spectrograms and save them into the database file
    dbi.create_database(output_file=output_db_name,  # empty brackets
                        dataset_name='train', selections=selections_train, data_dir=data_folder,
                        audio_repres=spec_cfg)

    dbi.create_database(output_file=output_db_name,
                        dataset_name='validation', selections=selections_val, data_dir=data_folder,
                        audio_repres=spec_cfg)

    dbi.create_database(output_file=output_db_name,
                        dataset_name='test', selections=selections_test, data_dir=data_folder,
                        audio_repres=spec_cfg)

    # create a new selection table for the test set - Fabio recommends for testing the model during development


def drop_out_of_bounds_sel(sel_table, file_durations):
    print('The length of df before dropping is ' + str(len(sel_table)))
    # add step here to drop selections past the end of the file
    # train

    # undo the multiindex for easier working
    sel_table = sel_table.reset_index(level=['filename', 'sel_id'])
    drop_rows_after = []
    drop_rows_before = []

    for idex, row in sel_table.iterrows():

        # filename is row[0], end time is idex.end
        index = file_durations.loc[file_durations['filename'] == row.filename].index[0]
        duration = file_durations['duration'][index]

        if duration < row.end:
            # drop the row corresponding to that sel_id and filename from the dataframe
            drop_rows_after.append(idex)

    print('The number of rows to drop is ' + str(len(drop_rows_after)))
    sel_table = sel_table.drop(drop_rows_after)

    print('The new length of sel table is ' + str(len(sel_table)))

    # remake the multiindex
    sel_table = sel_table.set_index(['filename', 'sel_id'])

    return sel_table

def train_classifier(database_h5, recipe, batch_size, n_epochs, output_name, spectro_file, checkpoint_folder, log_folder):
    """

    :param database_h5:
    :param recipe:
    :param batch_size:
    :param n_epochs:
    :param output_name:
    :param spectro_file:
    :return:
    """

    np.random.seed(1000)
    tf.random.set_seed(2000)

    db = dbi.open_file(database_h5, 'r')

    train_data = dbi.open_table(db, "/train/data")
    val_data = dbi.open_table(db, "/validation/data")

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

    resnet.save(output_name, audio_repr_file=spectro_file)


def create_detector(model_file, temp_model_folder, spec_folder, threshold, audio_folder, detections_csv, step_size, batch_size,
                    buffer):

    model = ResNetInterface.load(model_file=model_file, new_model_folder=temp_model_folder)

    audio_repr = load_audio_representation(path=spec_folder)

    spec_config = audio_repr['spectrogram']

    audio_loader = AudioFrameLoader(path=audio_folder, duration=spec_config['duration'],
                                    step=step_size, stop=False, representation=spec_config['type'],
                                    representation_params=spec_config, pad=False)

    detections = pd.DataFrame()

    batch_generator = batch_load_audio_file_data(loader=audio_loader, batch_size=batch_size)

    for batch_data in batch_generator:
        # Run the model on the spectrogram data from the current batch
        batch_predictions = model.run_on_batch(batch_data['data'], return_raw_output=True)

        # Lets store our data in a dictionary
        raw_output = {'filename': batch_data['filename'], 'start': batch_data['start'], 'end': batch_data['end'],
                      'score': batch_predictions}

        batch_detections = filter_by_threshold(raw_output, threshold=threshold)

        # What do these labels represent? Is it 0 for no, and 1 for yes? why is 0 included in the
        detections = pd.concat([detections, batch_detections], ignore_index=True)

    # detections_filtered = filter_by_label(detections, labels=1).reset_index(drop=True)

    #detections_grp = merge_overlapping_detections(detections_filtered)

    #print('The number of detections after filtering is ' + str(len(detections_grp)))

    #detections_grp.to_csv(detections_csv, index=False)
    detections.to_csv(detections_csv, index=False)


