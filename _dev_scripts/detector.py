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
    predictions = np.where(scores >= threshold, 1,0)

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

    # Manual dataset composition
    ulu_vals = [634, 179, 88]
    ulu2022_vals = [949, 274, 139]
    kk_vals = [1230, 348, 171]
    cb_vals = [130, 37, 18]

    ## Create Database ##

    # negatives tables and standarize for ketos
    ulu_neg = pd.read_excel(neg_folder + '\\' + 'std_ULU_negatives-manual-FINAL.xlsx')
    ulu_neg = ulu_neg.ffill()
    ulu_neg = sl.standardize(table=ulu_neg)
    print('Negatives standardized? ' + str(sl.is_standardized(ulu_neg)))

    ulu2022_neg = pd.read_excel(neg_folder + '\\' + 'std_ULU2022_negatives-manual-FINAL.xlsx')
    ulu2022_neg = ulu2022_neg.ffill()
    ulu2022_neg = sl.standardize(table=ulu2022_neg)
    print('Negatives standardized? ' + str(sl.is_standardized(ulu2022_neg)))

    kk_neg = pd.read_excel(neg_folder + '\\' + 'std_KK_negatives-manual-FINAL.xlsx')
    kk_neg = kk_neg.ffill()
    kk_neg = sl.standardize(table=kk_neg)
    print('Negatives standardized? ' + str(sl.is_standardized(kk_neg)))

    cb_neg = pd.read_excel(neg_folder + '\\' + 'std_CB_negatives-manual-FINAL.xlsx')
    cb_neg = cb_neg.ffill()
    cb_neg = sl.standardize(table=cb_neg)
    print('Negatives standardized? ' + str(sl.is_standardized(cb_neg)))

    # positives tables
    ulu_pos = pd.read_excel(pos_folder + '\\' + 'std_ULU_positives.xlsx')
    ulu_pos = ulu_pos.ffill()
    ulu_pos = sl.standardize(table=ulu_pos, start_labels_at_1=True)
    print('Positives standardized? ' + str(sl.is_standardized(ulu_pos)))

    ulu2022_pos = pd.read_excel(pos_folder + '\\' + 'std_ULU2022_positives.xlsx')
    ulu2022_pos = ulu2022_pos.ffill()
    ulu2022_pos = sl.standardize(table=ulu2022_pos, start_labels_at_1=True)
    print('Positives standardized? ' + str(sl.is_standardized(ulu2022_pos)))

    kk_pos = pd.read_excel(pos_folder + '\\' + 'std_KK_positives.xlsx')
    kk_pos = kk_pos.ffill()
    kk_pos = sl.standardize(table=kk_pos, start_labels_at_1=True)
    print('Positives standardized? ' + str(sl.is_standardized(kk_pos)))

    cb_pos = pd.read_excel(pos_folder + '\\' + 'std_CB_positives.xlsx')
    cb_pos = cb_pos.ffill()
    cb_pos = sl.standardize(table=cb_pos, start_labels_at_1=True)
    print('Positives standardized? ' + str(sl.is_standardized(cb_pos)))

    # join into complete tables

    ulu_pos_tr = ulu_pos.head(ulu_vals[0])
    ulu_pos_va = ulu_pos[ulu_vals[0]:ulu_vals[0] + ulu_vals[1]]
    ulu_pos_te = ulu_pos.tail(ulu_vals[2])

    ulu_neg_tr = ulu_neg.head(ulu_vals[0])
    ulu_neg_va = ulu_neg[ulu_vals[0]:ulu_vals[0] + ulu_vals[1]]
    ulu_neg_te = ulu_neg.tail(ulu_vals[2])

    ulu_tr = pd.concat([ulu_pos_tr, ulu_neg_tr])
    ulu_va = pd.concat([ulu_pos_va, ulu_neg_va])
    ulu_te = pd.concat([ulu_pos_te, ulu_neg_te])

    ulu2022_pos_tr = ulu2022_pos.head(ulu2022_vals[0])
    ulu2022_pos_va = ulu2022_pos[ulu2022_vals[0]:ulu2022_vals[0] + ulu2022_vals[1]]
    ulu2022_pos_te = ulu2022_pos.tail(ulu2022_vals[2])

    ulu2022_neg_tr = ulu2022_neg.head(ulu2022_vals[0])
    ulu2022_neg_va = ulu2022_neg[ulu2022_vals[0]:ulu2022_vals[0] + ulu2022_vals[1]]
    ulu2022_neg_te = ulu2022_neg.tail(ulu2022_vals[2])

    ulu2022_tr = pd.concat([ulu2022_pos_tr, ulu2022_neg_tr])
    ulu2022_va = pd.concat([ulu2022_pos_va, ulu2022_neg_va])
    ulu2022_te = pd.concat([ulu2022_pos_te, ulu2022_neg_te])

    kk_pos_tr = kk_pos.head(kk_vals[0])
    kk_pos_va = kk_pos[kk_vals[0]:kk_vals[0] + kk_vals[1]]
    kk_pos_te = kk_pos.tail(kk_vals[2])

    kk_neg_tr = kk_neg.head(kk_vals[0])
    kk_neg_va = kk_neg[kk_vals[0]:kk_vals[0] + kk_vals[1]]
    kk_neg_te = kk_neg.tail(kk_vals[2])

    kk_tr = pd.concat([kk_pos_tr, kk_neg_tr])
    kk_va = pd.concat([kk_pos_va, kk_neg_va])
    kk_te = pd.concat([kk_pos_te, kk_neg_te])

    cb_pos_tr = cb_pos.head(cb_vals[0])
    cb_pos_va = cb_pos[cb_vals[0]:cb_vals[0] + cb_vals[1]]
    cb_pos_te = cb_pos.tail(cb_vals[2])

    cb_neg_tr = cb_neg.head(cb_vals[0])
    cb_neg_va = cb_neg[cb_vals[0]:cb_vals[0] + cb_vals[1]]
    cb_neg_te = cb_neg.tail(cb_vals[2])

    cb_tr = pd.concat([cb_pos_tr, cb_neg_tr])
    cb_va = pd.concat([cb_pos_va, cb_neg_va])
    cb_te = pd.concat([cb_pos_te, cb_neg_te])

    # final three tables

    train = pd.concat([ulu_tr, ulu2022_tr, cb_tr, kk_tr])
    val = pd.concat([ulu_va, ulu2022_va, cb_va, kk_va])
    test = pd.concat([ulu_te, ulu2022_te, cb_te, kk_te])

    train.to_excel(main_folder + '\\' + 'sel_train.xlsx')
    val.to_excel(main_folder + '\\' + 'sel_val.xlsx')
    test.to_excel(main_folder + '\\' + 'sel_test.xlsx')

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

    ## Train Classifier ##
    # Set the random seeds for numpy and tensorflow for reproducibility
    np.random.seed(1000)
    tf.random.set_seed(2000)

    # Set the batch size and number of epochs for training
    batch_size = 16
    n_epochs = 40
    log_folder = main_folder + '\\' + 'logs'
    checkpoint_folder = main_folder + '\\' + 'checkpoints'

    # Open the database
    db = dbi.open_file(db_name, 'r')

    # Open the training and validation tables from the database
    train_data = dbi.open_table(db, "/train/data")
    val_data = dbi.open_table(db, "/val/data")

    #
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
    resnet.save(output_name, audio_repr_file=spec_file)

    log_folder = main_folder + '\\' + 'logs'

    log_file = pd.read_csv(log_folder + '\\' + 'log.csv')

    tr_results = log_file[log_file['dataset']=='train']
    va_results = log_file[log_file['dataset']=='val']

    sns.lineplot(data=tr_results, x='epoch', y='loss', label='train', legend='auto')
    sns.lineplot(data=va_results, x='epoch', y='loss', label='val', legend='auto')


    test_filled = test.reset_index(allow_duplicates=True)

    audio_folder = main_folder + '\\' + 'audio'

    for idex, row in test_filled.iterrows():
        shutil.copyfile(test_filled.loc[idex]['filename'], audio_folder + '\\' + test_filled.loc[idex]['filename'].split('\\')[-1])

    print('done')

    temp_folder = main_folder + '\\' + 'ringedS_tmp_folder'
    detections_csv = main_folder + '\\' + 'detections_raw.csv'
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

        # What do these labels represent? Is it 0 for no, and 1 for yes? why is 0 included in the
        detections = pd.concat([detections, batch_detections], ignore_index=True)

    detections.to_csv(detections_csv, index=False)

    output_dir = main_folder + '\\' + 'metrics'

    db = dbi.open_file(db_name, 'r')

    # Load the trained model
    model = ResNetInterface.load(output_name, load_audio_repr=False, new_model_folder=temp_folder)

    # Open the table in the database at the root level
    table = dbi.open_table(db, '/test')

    # Convert the data to the correct format for the model, and generate batches of data
    gens = []

    # not sure?
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

