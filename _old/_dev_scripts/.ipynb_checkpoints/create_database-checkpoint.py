import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import shutil
import os
import csv
from ketos.data_handling import selection_table as sl
import ketos.data_handling.database_interface as dbi
from ketos.data_handling.parsing import load_audio_representation
from ketos.data_handling.data_feeding import BatchGenerator, JointBatchGen
from ketos.neural_networks.resnet import ResNetInterface
from ketos.audio.audio_loader import AudioFrameLoader, AudioLoader, SelectionTableIterator
from ketos.audio.spectrogram import MagSpectrogram
from ketos.neural_networks.dev_utils.detection import batch_load_audio_file_data, filter_by_threshold
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":

    # User inputs
    main_folder = r'C:\Users\kzammit\Documents\Detector\manual_detector_2sec'
    neg_folder = main_folder + '\\' + r'inputs\annotations\edited_sels\negatives'
    pos_folder = main_folder + '\\' + r'inputs\annotations\edited_sels\positives'
    spec_file = main_folder + '\\' + r'inputs\spec_config_2sec.json'
    data_folder = r'D:\ringed-seal-data'
    db_name = main_folder + '\\' r'manual_database_2sec_tes3.h5'
    recipe = main_folder + '\\' + r'inputs\resnet_recipe.json'
    output_name = main_folder + '\\' + 'rs-2sec.kt'

    train = pd.read_excel(main_folder + '\\' + 'sel_train.xlsx')
    train = train.ffill()
    train = train.drop(["Unnamed: 0", "annot_id.1"], axis=1)
    train = sl.standardize(table=train)

    val = pd.read_excel(main_folder + '\\' + 'sel_val.xlsx')
    val = val.ffill()
    val = val.drop(["Unnamed: 0", "annot_id.1"], axis=1)
    val = sl.standardize(table=val)
    
    test = pd.read_excel(main_folder + '\\' + 'sel_test.xlsx')
    test = test.ffill()
    test = test.drop(["Unnamed: 0", "annot_id.1"], axis=1)
    test = sl.standardize(table=test)

    # Join into a database
    # Use these spectrogram parameters
    spec_cfg = load_audio_representation(spec_file, name="spectrogram")

    dbi.create_database(output_file=db_name,  # empty brackets
                        dataset_name=r'train', selections=train, data_dir=data_folder,
                        audio_repres=spec_cfg)

    dbi.create_database(output_file=db_name,  # empty brackets
                        dataset_name=r'val', selections=val, data_dir=data_folder,
                        audio_repres=spec_cfg)

    dbi.create_database(output_file=db_name,  # empty brackets
                        dataset_name=r'test', selections=test, data_dir=data_folder,
                        audio_repres=spec_cfg)

