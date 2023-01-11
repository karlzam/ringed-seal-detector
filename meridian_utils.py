import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from ketos.data_handling import selection_table as sl
import ketos.data_handling.database_interface as dbi
from ketos.data_handling.parsing import load_audio_representation
from ketos.data_handling.parsing import load_audio_representation
from ketos.data_handling.data_feeding import BatchGenerator
from ketos.audio.spectrogram import MagSpectrogram
from ketos.neural_networks.resnet import ResNetInterface
import ketos.neural_networks.dev_utils.detection as det
from ketos.audio.spectrogram import MagSpectrogram
from ketos.audio.audio_loader import AudioFrameLoader
from ketos.neural_networks.dev_utils.detection import process, save_detections


def create_database(train_csv, val_csv, length, output_db_name, spectro_file, data_folder):
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

    # standardize the training and validation annotation csv files to ketos format
    std_annot_train = sl.standardize(table=annot_train, labels=["bark"], start_labels_at_1=True, trim_table=True)
    print('Training data standardized? ' + str(sl.is_standardized(std_annot_train)))
    std_annot_val = sl.standardize(table=annot_val, labels=["bark"], start_labels_at_1=True, trim_table=True)
    print('Validation data standardized? ' + str(sl.is_standardized(std_annot_val)))

    # create segments of uniform length from the annotations tables
    positives_train = sl.select(annotations=std_annot_train, length=length, step=0.5, min_overlap=0.5, center=False)
    positives_val = sl.select(annotations=std_annot_val, length=length, step=0.0, center=False)

    # calculate the file durations
    #TODO: Update this to be able to look at the actual table, get the pathways to each file, and check there
    file_durations_train = sl.file_duration_table(data_folder + '\\' + 'train')
    file_durations_val = sl.file_duration_table(data_folder + '\\' + 'val')

    # generate negative segments (the same number as the positive segments),
    # specifying the same length as the training data
    negatives_train = sl.create_rndm_selections(annotations=std_annot_train, files=file_durations_train,
                                                length=length, num=len(positives_train), trim_table=True)
    negatives_val = sl.create_rndm_selections(annotations=std_annot_val, files=file_durations_val,
                                              length=length, num=len(positives_val), trim_table=True)

    # join the positive and negative vals together
    selections_train = pd.concat([positives_train, negatives_train], sort=False)
    selections_val = pd.concat([positives_val, negatives_val], sort=False)

    # load in the spectrogram settings
    spec_cfg = load_audio_representation(spectro_file, name="spectrogram")

    # TODO: Update this to be able to look at multiple data dirs instead of a train and validation dir
    # compute spectrograms and save them into the database file
    dbi.create_database(output_file=output_db_name, data_dir=data_folder + '\\' + 'train',
                        dataset_name='train', selections=selections_train,
                        audio_repres=spec_cfg)
    dbi.create_database(output_file=output_db_name, data_dir=data_folder + '\\' + 'val',
                        dataset_name='validation', selections=selections_val,
                        audio_repres=spec_cfg)


def train_classifier(database_h5, recipe, batch_size, n_epochs, output_name, spectro_file, checkpoint_folder):
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
                                   shuffle=True, refresh_on_epoch_end=False)

    resnet = ResNetInterface.build_from_recipe_file(recipe)

    resnet.train_generator = train_generator
    resnet.val_generator = val_generator

    resnet.checkpoint_dir = checkpoint_folder

    resnet.train_loop(n_epochs=n_epochs, verbose=True)

    db.close()

    resnet.save_model(output_name, audio_repr=spectro_file)


def create_detector(model_file, temp_model_folder, threshold, audio_folder, detections_csv, step_size, batch_size,
                    buffer):
    model, audio_repr = ResNetInterface.load_model_file(model_file=model_file, new_model_folder=temp_model_folder,
                                                        load_audio_repr=True)

    spec_config = audio_repr[0]['spectrogram']

    audio_loader = AudioFrameLoader(path=audio_folder, duration=spec_config['duration'],
                                    step=step_size, stop=False, representation=MagSpectrogram,
                                    representation_params=spec_config)

    detections = process(audio_loader, model=model, batch_size=batch_size, progress_bar=True,
                         group=True, threshold=threshold, buffer=buffer)

    save_detections(detections=detections, save_to=detections_csv)
