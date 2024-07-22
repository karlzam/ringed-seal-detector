import pandas as pd
import numpy as np
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def compute_detections(labels, scores, threshold):
    # Compute the positive scores above threshold, 1 if it is above threshold, 0 if it is not
    predictions = np.where(scores >= threshold, 1, 0)

    # TP: Does the annotated label match the prediction above threshold? Bc "scores" is defined as the positive threshold, this represents TP
    TP = tf.math.count_nonzero(predictions * labels).numpy()

    # TN: Negative score is "predictions - 1" bc predictions was for the positive result, labels-1 so that the negatives are multiplied by 1
    TN = tf.math.count_nonzero((predictions - 1) * (labels - 1)).numpy()

    # And so on
    FP = tf.math.count_nonzero(predictions * (labels - 1)).numpy()
    FN = tf.math.count_nonzero((predictions - 1) * labels).numpy()

    return predictions, TP, TN, FP, FN


folder = r'E:\baseline-with-normalization-reduce-tonal\ulu2023\metrics\ensemble\class'

files = glob.glob(folder + '\*.csv')

thresholds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
              0.95, 1]

all = []

for idx, file in enumerate(files):

    class_file = pd.read_csv(file)

    for threshold in thresholds:

        temp = class_file[class_file['threshold'] == threshold]

        predicted, TP, TN, FP, FN = compute_detections(temp['label'], temp['score'], threshold)

        model_name = file.split("\\")[-1].split('.')[0]
        temp_array = [model_name, threshold, TP, TN, FP, FN]

        all.append(temp_array)

df = pd.DataFrame(all, columns=["model", "threshold", "TP", "TN", "FP", "FN"])

df_calcs_edited = df[df['threshold'] != 0]
df_calcs_edited = df_calcs_edited[df_calcs_edited['threshold'] != 1]

df_mean = df_calcs_edited[df_calcs_edited['model']=='classifications-mean']
df_90 = df_calcs_edited[df_calcs_edited['model']=='classifications-90']
df_10 = df_calcs_edited[df_calcs_edited['model']=='classifications-10']

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.set_title("Average Counts for 10 Model Runs with Shaded Standard Deviation")

ax1.plot(df_mean['threshold'], df_mean['TP'], '#377eb8', label='TP')
ax1.fill_between(df_mean['threshold'], df_10['TP'], df_90['TP'], color='#377eb8', alpha=0.2)

ax1.plot(df_mean['threshold'], df_mean['TN'], '#ff7f00', label='TN')
ax1.fill_between(df_mean['threshold'], df_10['TN'], df_90['TN'], color='#ff7f00', alpha=0.2)

ax1.set_xlabel("Threshold")
ax1.set_ylabel("Number")
ax1.legend()

ax2.plot(df_mean['threshold'], df_mean['FP'], '#4daf4a', label='FP')
ax2.fill_between(df_mean['threshold'], df_10['FP'], df_90['FP'], color='#4daf4a', alpha=0.2)

ax2.plot(df_mean['threshold'], df_mean['FN'], '#999999', label='FN')
ax2.fill_between(df_mean['threshold'], df_10['FN'], df_90['FN'], color='#999999', alpha=0.2)
ax2.set_xlabel("Threshold")
ax2.set_ylabel("Number")
ax2.legend()

plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
plt.tight_layout()
# plt.show()
plt.savefig(folder + '\\' + 'average_counts.png')

## Plot Confusion Matrix

df_confs = df_mean[df_mean['threshold']==0.55]

from mlxtend.plotting import plot_confusion_matrix

# Your Confusion Matrix
cm = np.array([[round(float(df_confs['TP']), 2), round(float(df_confs['FP']), 2)],
               [round(float(df_confs['FN']), 2), round(float(df_confs['TN']), 2)]])

# Classes
classes = ['RS', 'O']

figure2, ax = plot_confusion_matrix(conf_mat = cm,
                                   class_names = classes,
                                   show_absolute = True,
                                   show_normed = True,
                                   colorbar = True)

plt.tight_layout()
plt.savefig(folder + '\\' + 'confusion_matrix_avg.png')


