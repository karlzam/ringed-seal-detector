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

# Compute metrics
def compute_detections(labels, scores, threshold=0.5):
    """

    :param labels:
    :param scores:
    :param threshold:
    :return:
    """
    predictions = np.where(scores >= threshold, 1, 0)

    TP = tf.math.count_nonzero(predictions * labels).numpy()
    TN = tf.math.count_nonzero((predictions - 1) * (labels - 1)).numpy()
    FP = tf.math.count_nonzero(predictions * (labels - 1)).numpy()
    FN = tf.math.count_nonzero((predictions - 1) * labels).numpy()

    return predictions, TP, TN, FP, FN


if __name__ == "__main__":

    # User inputs
    main_folder = r'C:\Users\kzammit\Documents\Detector\manual_detector_2sec'
    neg_folder = main_folder + '\\' + r'inputs\annotations\edited_sels\negatives'
    pos_folder = main_folder + '\\' + r'inputs\annotations\edited_sels\positives'
    spec_file = main_folder + '\\' + r'inputs\spec_config_2sec.json'
    data_folder = r'D:\ringed-seal-data'
    db_name = main_folder + '\\' r'manual_database_2sec.h5'
    recipe = main_folder + '\\' + r'inputs\resnet_recipe.json'
    output_name = main_folder + '\\' + 'rs-2sec.kt'
    temp_folder = main_folder + '\\' + 'ringedS_tmp_folder'
    detections_csv = main_folder + '\\' + 'detections_raw.csv'
    audio_folder = main_folder + '\\' + 'audio'

    threshold = 0.5
    step_size = 2.0
    batch_size = 16
    buffer = 0.5

    model = ResNetInterface.load(model_file=output_name, new_model_folder=temp_folder)

    audio_repr = load_audio_representation(path=spec_file)

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

        detections = pd.concat([detections, batch_detections], ignore_index=True)

    detections.to_csv(detections_csv, index=False)

    output_dir = main_folder + '\\' + 'metrics'

    db = dbi.open_file(db_name, 'r')

    # Load the trained model
    model = ResNetInterface.load(output_name, load_audio_repr=False, new_model_folder=temp_folder)

    # Open the table in the database at the root level
    table = dbi.open_table(db, '/test')

    gens = []

    batch_size = int(batch_size / sum(1 for _ in db.walk_nodes(table, "Table")))

    # for the testing dataset
    for group in db.walk_nodes(table, "Table"):
        generator = BatchGenerator(batch_size=batch_size, data_table=group,
                                   output_transform_func=ResNetInterface.transform_batch, shuffle=False,
                                   refresh_on_epoch_end=False, x_field='data', return_batch_ids=True)

        # attach the batches together so there's one for each dataset
        gens.append(generator)

    gen = JointBatchGen(gens, n_batches='min', shuffle_batch=False, reset_generators=False, return_batch_ids=True)

    scores = []
    labels = []

    for batch_id in range(gen.n_batches):
        hdf5_ids, batch_X, batch_Y = next(gen)

        batch_labels = np.argmax(batch_Y, axis=1)

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

