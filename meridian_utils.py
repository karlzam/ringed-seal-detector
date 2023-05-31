import pandas as pd
import numpy as np
import tensorflow as tf
import os
from ketos.data_handling import selection_table as sl
import ketos.data_handling.database_interface as dbi
from ketos.data_handling.parsing import load_audio_representation
from ketos.data_handling.data_feeding import BatchGenerator
from ketos.neural_networks.resnet import ResNetInterface
from ketos.audio.spectrogram import MagSpectrogram
from ketos.audio.audio_loader import AudioFrameLoader, AudioLoader, SelectionTableIterator
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
    std_annot_train = sl.standardize(table=annot_train, labels=["B", "BY"], start_labels_at_1=True, trim_table=True)

    # force labels all to be 1 for binary classification
    std_annot_train['label'] = std_annot_train['label'].replace(2, 1)
    print('Remember youre forcing all labels to be 1! Training data standardized? ' + str(sl.is_standardized(std_annot_train)))

    std_annot_val = sl.standardize(table=annot_val, labels=["B", "BY"], start_labels_at_1=True, trim_table=True)
    std_annot_val['label'] = std_annot_val['label'].replace(2, 1)
    print('Remember youre forcing all labels to be 1! Validation data standardized? ' + str(sl.is_standardized(std_annot_val)))

    # create segments of uniform length from the annotations tables
    positives_train = sl.select(annotations=std_annot_train, length=length, step=0.5, min_overlap=0.5, center=False)
    positives_val = sl.select(annotations=std_annot_val, length=length, step=0.0, center=False)

    '''
    # calculate the file durations
    # file_durations_val = sl.file_duration_table(data_folder)
    # file_durations_train = sl.file_duration_table(data_folder)

    folders = [f.path for f in os.scandir(data_folder) if f.is_dir()]

    # temp fix to get around corrupt ULU files
    #folders = folders[0:6]

    # moving onto corrupt ULU files
    #folders = folders[6:]

    file_durations = pd.DataFrame()
    for folder in folders:

        folder_name = folder.split('\\')[-1]

        # get the durations for all the files within this folder
        file_durations_for_folder = sl.file_duration_table(data_folder + "\\" + str(folder_name))

        # it only keeps the wav filename, need to append the path back onto it I think? Let's try that
        file_durations_for_folder['fixed_filename'] = folder + "\\" + file_durations_for_folder['filename']

        # drop the og filename column
        file_durations_for_folder = file_durations_for_folder.drop(['filename'], axis=1)

        # rename the appended filename
        file_durations_for_folder = file_durations_for_folder.rename(columns={'fixed_filename': 'filename'})

        #file_durations = file_durations.append(sl.file_duration_table(data_folder + "\\" + str(folder_name)))
        file_durations = file_durations.append(file_durations_for_folder)

    #print(len(file_durations))
    #file_durations.to_excel('Ulu_durations.xlsx', index=False)

    #file_durations.to_excel('file_durations_minusUlu.xlsx', index=False)
    
    '''

    file_durations = pd.read_excel(r'C:\Users\kzammit\Documents\Detector\20230531\inputs\all_file_durations.xlsx')

    # generate negative segments (the same number as the positive segments),
    # specifying the same length as the training data
    # Q for Fabio: It's ok that I'm using one big file durations file right? Not split into train and val?
    # the tutorial has it split into train and val
    negatives_train = sl.create_rndm_selections(annotations=std_annot_train, files=file_durations,
                                                length=length, num=len(positives_train), trim_table=True)


    negatives_val = sl.create_rndm_selections(annotations=std_annot_val, files=file_durations,
                                              length=length, num=len(positives_val), trim_table=True)

    # join the positive and negative vals together
    selections_train = pd.concat([positives_train, negatives_train], sort=False)
    #selections_train.to_excel('train_selections_20230530.xlsx')
    selections_val = pd.concat([positives_val, negatives_val], sort=False)
    #selections_val.to_excel('val_selections_20230530.xlsx')

    # load in the spectrogram settings
    spec_cfg = load_audio_representation(spectro_file, name="spectrogram")

    # compute spectrograms and save them into the database file
    dbi.create_database(output_file=output_db_name, # empty brackets
                        dataset_name='train', selections=selections_train, data_dir=data_folder,
                        audio_repres=spec_cfg)

    dbi.create_database(output_file=output_db_name,
                        dataset_name='validation', selections=selections_val, data_dir=data_folder,
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

    #gpu = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(gpu, True)

    model, audio_repr = ResNetInterface.load_model_file(model_file=model_file, new_model_folder=temp_model_folder,
                                                        load_audio_repr=True)

    spec_config = audio_repr[0]['spectrogram']

    audio_loader = AudioFrameLoader(path=audio_folder, duration=spec_config['duration'],
                                    step=step_size, stop=False, representation=MagSpectrogram,
                                    representation_params=spec_config)

    detections = process(audio_loader, model=model, batch_size=batch_size, progress_bar=True,
                         group=True, threshold=threshold, buffer=buffer)

    save_detections(detections=detections, save_to=detections_csv)
