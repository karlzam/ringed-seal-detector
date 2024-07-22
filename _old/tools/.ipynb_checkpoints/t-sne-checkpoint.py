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
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

from ketos.data_handling import selection_table as sl
import ketos.data_handling.database_interface as dbi
from ketos.data_handling.parsing import load_audio_representation
from ketos.data_handling.data_feeding import BatchGenerator
from ketos.neural_networks.resnet import ResNetInterface
from ketos.audio.audio_loader import AudioFrameLoader, AudioLoader, SelectionTableIterator
from ketos.audio.spectrogram import MagSpectrogram
from ketos.neural_networks.dev_utils.detection import batch_load_audio_file_data, filter_by_threshold, merge_overlapping_detections, filter_by_label
from ketos.data_handling.data_feeding import JointBatchGen

def feature_extractor(pre_trained_model):
    extractor = tf.keras.models.Sequential(pre_trained_model.model.layers[0:4])
    extractor.trainable = False
    return extractor


thresholds = [0.4]
step_size = 1.0
batch_size = 16
buffer = 0.5

output_dir = r'E:\final-baseline-detector\metrics'
main_folder = r'E:\final-baseline-detector'
db_name = main_folder + '\\' + 'final-baseline-db.h5'
model_name = main_folder + '\\' + 'final-baseline-model.kt'
temp_folder = main_folder + '\\' + 'temp'

db = dbi.open_file(db_name, 'r')
table = dbi.open_table(db, '/test')
classification_csv = "classifications.csv"
metric_csv = "metrics.csv"
stats_csv = "stats.csv"

model = ResNetInterface.load(model_name, load_audio_repr=False, new_model_folder=temp_folder)
gens = []
batch_size = int(batch_size / sum(1 for _ in db.walk_nodes(table, "Table")))

for group in db.walk_nodes(table, "Table"):
    # Create a batch generator for this table
    generator = BatchGenerator(batch_size=batch_size, data_table=group,
                               output_transform_func=ResNetInterface.transform_batch, shuffle=False,
                               refresh_on_epoch_end=False, x_field='data', return_batch_ids=True)

    # Append the generator to the gens array
    gens.append(generator)

gen = JointBatchGen(gens, n_batches='min', shuffle_batch=False, reset_generators=False, return_batch_ids=True)
simplified = feature_extractor(model)

scores = []
labels = []
output = []

for batch_id in range(gen.n_batches):
    # Get the ids, spectrograms, and labels for the data in the batch
    hdf5_ids, batch_X, batch_Y = next(gen)

    # Get the labels for the batch data, using the "argmax" func which returns the col header, so 0 is a noise segment, 1 is a rs segment
    batch_labels = np.argmax(batch_Y, axis=1)

    # Returns the scores for the batch for the "positive" class - this is used in the compute detections function later on
    batch_scores = model.model.predict_on_batch(batch_X)[:, 1]

    # Add these scores for this batch to the overall list
    scores.extend(batch_scores)
    labels.extend(batch_labels)

    # get array for each of size (64,)
    output.extend(simplified(batch_X))

output_numpy = [x.numpy() for x in output]
df = pd.DataFrame(output_numpy)

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df)
tsne_df = pd.DataFrame(columns=['CP1', 'CP2', 'target'], data=np.column_stack((tsne_results, labels)))
grid = sns.FacetGrid(tsne_df, hue="target")
grid.map(plt.scatter, 'CP1', 'CP2', alpha=0.2).add_legend()
plt.savefig('tsne-coloured.png')

tsne_3d = TSNE(n_components=3, random_state=0)
projections = tsne_3d.fit_transform(df)
tsne_df3d = pd.DataFrame(columns=['CP1', 'CP2', 'CP3', 'target'], data=np.column_stack((projections, labels)))

fig = plt.figure(figsize=(6,6))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

# get colormap from seaborn
#cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())

# plot
sc = ax.scatter(tsne_df3d['CP1'], tsne_df3d['CP2'], tsne_df3d['CP3'], s=40, c=tsne_df3d['target'], marker='o', cmap='viridis', alpha=0.5)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# legend
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

# save
plt.savefig("scatter_hue.png", bbox_inches='tight')



