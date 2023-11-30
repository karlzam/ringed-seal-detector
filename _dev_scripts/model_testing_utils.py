import ketos.audio
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from ketos.data_handling import selection_table as sl
import ketos.data_handling.database_interface as dbi
from ketos.data_handling.data_feeding import BatchGenerator, JointBatchGen
from ketos.neural_networks.resnet import ResNetInterface


def compare(annotations, detections):
    """

    :param annotations:
    :param detections:
    :return:
    """

    detected_list = []

    annotations = pd.read_csv(annotations)
    detections = pd.read_csv(detections)

    for idx, row in annotations.iterrows():  # loop over annotations
        filename_annot = row['filename'].split("\\")[-1]
        time_annot_start = row['start']
        time_annot_end = row['end']
        detected = False
        for _, d in detections.iterrows():  # loop over detections
            filename_det = d['filename']
            start_det = d['start']
            end_det = start_det + d['end']
            # if the filenames match and the annotated time falls with the start and
            # end time of the detection interval, consider the call detected
            if filename_annot == filename_det and time_annot_start >= start_det and time_annot_end <= end_det:
                detected = True
                break

        detected_list.append(detected)

    annotations['detected'] = detected_list  # add column to the annotations table

    return annotations


def compute_detections(labels, scores, threshold=0.5):
    """

    :param labels:
    :param scores:
    :param threshold:
    :return:
    """
    predictions = np.where(scores >= threshold, 1,0)

    TP = tf.math.count_nonzero(predictions * labels).numpy()
    TN = tf.math.count_nonzero((predictions - 1) * (labels - 1)).numpy()
    FP = tf.math.count_nonzero(predictions * (labels - 1)).numpy()
    FN = tf.math.count_nonzero((predictions - 1) * labels).numpy()

    return predictions, TP, TN, FP, FN


def compute_model_outcome(model, db, new_model_folder, output_dir, batch_size, threshold, table_name):
    """

    :param model:
    :param db:
    :param new_model_folder:
    :param output_dir:
    :param batch_size:
    :param threshold:
    :param table_name:
    :return:
    """

    # Open the database file
    db = dbi.open_file(db, 'r')

    # Load the trained model
    model = ResNetInterface.load(model, load_audio_repr=False, new_model_folder=new_model_folder)

    # The root name in the database is slash
    # table_name = '/test'

    # Open the table in the database at the root level
    table = dbi.open_table(db, table_name)

    # Convert the data to the correct format for the model, and generate batches of data
    gens = []

    # not sure?
    batch_size = int(batch_size / sum(1 for _ in db.walk_nodes(table, "Table")))

    # for each dataset (I think?), ie. train/test/val
    for group in db.walk_nodes(table, "Table"):
        generator = BatchGenerator(batch_size=batch_size, data_table=group,
                                   output_transform_func=ResNetInterface.transform_batch, shuffle=False,
                                   refresh_on_epoch_end=False, x_field='data', return_batch_ids=True)

        # attach the batches together? so there's one for each dataset
        gens.append(generator)

    # isn't this using the training/val/and testing data to compute these metrics? prolly not what you want to do?
    gen = JointBatchGen(gens, n_batches='min', shuffle_batch=False, reset_generators=False, return_batch_ids=True)

    scores = []
    labels = []

    for batch_id in range(gen.n_batches):
        hdf5_ids, batch_X, batch_Y = next(gen)

        batch_labels = np.argmax(batch_Y, axis=1)

        # will return the scores for just one class (with label 1)
        batch_scores = model.model.predict_on_batch(batch_X)[:, 1]

        scores.extend(batch_scores)
        labels.extend(batch_labels)

    labels = np.array(labels)
    scores = np.array(scores)

    print('Length of labels is ' + str(len(labels)))

    predicted, TP, TN, FP, FN = compute_detections(labels, scores, threshold)

    print(f'\nSaving detections output to {output_dir}/')

    df_group = pd.DataFrame()
    for group in db.walk_nodes(table, "Table"):
        df = pd.DataFrame({'id': group[:]['id'], 'filename': group[:]['filename']})
        df_group = pd.concat([df_group, df], ignore_index=True)
    df_group['label'] = labels[:]
    df_group['predicted'] = predicted[:]
    df_group['score'] = scores[:]
    df_group.to_csv(os.path.join(os.getcwd(), output_dir, "classifications.csv"), mode='w', index=False)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    FPP = FP / (TN + FP)
    confusion_matrix = [[TP, FN], [FP, TN]]
    print(f'\nPrecision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print('\nConfusionMatrix:')
    print('\n[TP, FN]')
    print('[FP, TN]')
    print(f'{confusion_matrix[0]}')
    print(f'{confusion_matrix[1]}')

    print(f"\nSaving metrics to {output_dir}/")

    # Saving precision recall and F1 Score for the defined thrshold
    metrics = {'Precision': [precision], 'Recall': [recall], "F1 Score": [f1]}
    metrics_df = pd.DataFrame(data=metrics)

    metrics_df.to_csv(os.path.join(os.getcwd(), output_dir, "metrics.csv"), mode='w', index=False)

    # Appending a confusion matrix to the file
    row1 = ["Confusion Matrix", "Predicted"]
    row2 = ["Actual", "RS", "Background Noise"]
    row3 = ["RS", TP, FN]
    row4 = ["Background Noise", FP, TN]
    with open(os.path.join(os.getcwd(), output_dir, "metrics.csv"), 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(row1)
        writer.writerow(row2)
        writer.writerow(row3)
        writer.writerow(row4)

    db.close()


def make_confusion_matrix(classifications_file, output_dir):
    """

    :param classifications_file:
    :param output_dir:
    :return:
    """

    classifications = pd.read_csv(classifications_file)
    cm = confusion_matrix(classifications['label'], classifications['predicted'])

    labels = ['True Pos', 'False Neg', 'False Pos', 'True Neg']
    categories = ['Ringed Seal', 'Noise']

    #confusion_matrix_plot(cm, output_dir, group_names=labels, categories=categories,
    #                 cmap=sns.diverging_palette(20, 220, as_cmap=True))

    confusion_matrix_plot(cm, output_dir, group_names=labels, categories=categories,
                     cmap='viridis')


def confusion_matrix_plot(cf, output_folder,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=True):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.

    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        #plt.xlabel('Predicted label' + stats_text)
        plt.xlabel('Predicted label')
    else:
        plt.xlabel(stats_text)

    if title:
        #plt.title(title)
        #plt.title(stats_text)
        print('no title')

    plt.savefig(output_folder + '\\' + 'confusion_matrix.png')


