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
from ketos.neural_networks.dev_utils.detection import batch_load_audio_file_data, filter_by_threshold, filter_by_label, merge_overlapping_detections
from ketos.data_handling.data_feeding import JointBatchGen

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

print('done importing packages')

pos = pd.read_excel(r'D:\ringed_seal_selection_tables\ulu2023\positives.xlsx')

pos = pos.ffill()

pos.to_csv(r'E:\detector_2\annots\pos\ULU2023_all_formatted_1sec.csv', index=False)

neg = pd.read_excel(r'D:\ringed_seal_selection_tables\ulu2023\negatives-manual.xlsx')

neg = neg.ffill()

neg.to_csv(r'E:\detector_2\annots\neg\ULU2023-negs.csv', index=False)


print('test')