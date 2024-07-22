import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import shutil
import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from ketos.data_handling import selection_table as sl
import ketos.data_handling.database_interface as dbi
from ketos.data_handling.parsing import load_audio_representation
from ketos.data_handling.data_feeding import BatchGenerator, JointBatchGen
from ketos.neural_networks.resnet import ResNetInterface
from ketos.audio.audio_loader import AudioFrameLoader, AudioLoader, SelectionTableIterator
from ketos.audio.spectrogram import MagSpectrogram
from ketos.neural_networks.dev_utils.detection import batch_load_audio_file_data, filter_by_threshold

model_name = r'C:\Users\kzammit\Documents\test\rs-model-5.kt'
temp_name = r'C:\Users\kzammit\Documents\test\rs-model-5-temp.kt'

model = ResNetInterface.load(model_file=model_name, new_model_folder=temp_name)

print('test')