import pandas as pd
import os
import tensorflow as tf
from ketos.data_handling.parsing import load_audio_representation
from ketos.neural_networks.resnet import ResNetInterface
from ketos.audio.audio_loader import AudioFrameLoader
from ketos.neural_networks.dev_utils.detection import batch_load_audio_file_data, filter_by_threshold, filter_by_label, merge_overlapping_detections
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

print('done importing packages')


def get_batch_generator(spectro_file, audio_folder, step_size, batch_size):
    audio_repr = load_audio_representation(path=spectro_file)

    spec_config = audio_repr['spectrogram']

    audio_loader = AudioFrameLoader(path=audio_folder, duration=spec_config['duration'],
                                    step=step_size, stop=False, representation=spec_config['type'],
                                    representation_params=spec_config, pad=False)

    batch_generator = batch_load_audio_file_data(loader=audio_loader, batch_size=batch_size)

    return batch_generator


def load_models(model_names, temp_folders):
    models = []
    for idx, model_name in enumerate(model_names):
        models.append(ResNetInterface.load(model_file=model_name, new_model_folder=temp_folders[idx]))

    return models


def get_detections(batch_generator, models, output_dir, threshold, max_sd, raven_txt, audio_folder):
    detections_pos = pd.DataFrame()
    detections_neg = pd.DataFrame()

    for ibx, batch_data in enumerate(batch_generator):

        for idx, model in enumerate(models):

            # Run the model on the spectrogram data from the current batch
            batch_predictions = model.run_on_batch(batch_data['data'], return_raw_output=True)

            if idx == 0:
                # Store data in a dictionary
                raw_output_neg = {'filename': batch_data['filename'], 'start': batch_data['start'],
                                  'end': batch_data['end'], '0-0': batch_predictions[:, 0]}

                raw_output_pos = {'filename': batch_data['filename'], 'start': batch_data['start'],
                                  'end': batch_data['end'], '1-0': batch_predictions[:, 1]}

            else:
                raw_output_neg |= {'0-' + str(idx): batch_predictions[:, 0]}
                raw_output_pos |= {'1-' + str(idx): batch_predictions[:, 1]}

        detections_pos = pd.concat([detections_pos, pd.DataFrame.from_dict(raw_output_pos)])
        detections_neg = pd.concat([detections_neg, pd.DataFrame.from_dict(raw_output_neg)])

    mean_cols_pos = detections_pos.columns[3:]
    mean_cols_neg = detections_neg.columns[3:]

    # Calculate central tendency (Median)
    detections_pos['med-pos'] = detections_pos[mean_cols_pos].quantile(0.5, axis=1)
    detections_neg['med-neg'] = detections_neg[mean_cols_neg].quantile(0.5, axis=1)

    # Calculate intra-ensemble standard deviation for positive tracks
    detections_pos['ensemble_sd'] = detections_pos[mean_cols_pos].std(axis=1)

    merge_df = detections_pos[['filename', 'start', 'end', 'med-pos', 'ensemble_sd']].copy()
    merge_df['med-neg'] = detections_neg['med-neg']

    scores = []
    for row in merge_df.iterrows():
        score = [row[1]['med-neg'], row[1]['med-pos']]
        scores.extend([score])

    merge_df['score'] = scores

    # Prepare data for Ketos baseline threshold filter
    dict_data = {'filename': merge_df['filename'], 'start': merge_df['start'], 'end': merge_df['end'], 'score': merge_df['score']}
    filter_detections = filter_by_threshold(dict_data, threshold=threshold)
    detections_filtered_initial = filter_by_label(filter_detections, labels=1).reset_index(drop=True)
    
    # NEW: Map the standard deviation back to filter out high-variance split-juries
    # We match using the original indices preserved or explicitly matching file/time coordinates
    merged_with_sd = detections_filtered_initial.merge(
        merge_df[['filename', 'start', 'end', 'ensemble_sd']], 
        left_on=['filename', 'start', 'end'], 
        right_on=['filename', 'start', 'end'], 
        how='left'
    )
    
    # Filter by the consensus gate constraint (SD < max_sd)
    detections_filtered = merged_with_sd[merged_with_sd['ensemble_sd'] < max_sd].drop(columns=['ensemble_sd']).reset_index(drop=True)

    print('The total number of initial threshold detections is ' + str(len(detections_filtered_initial)))
    print('The total number of high-consensus detections is ' + str(len(detections_filtered)))
    
    detections_grp = merge_overlapping_detections(detections_filtered)
    print('The total number of grouped detections is ' + str(len(detections_grp)))
    detections_grp.to_excel(output_dir + '\\' + 'detections-filtered-and-grouped.xlsx', index=False)

    results_table = detections_grp

    dir_path = os.path.dirname(os.path.realpath(__file__)) if '__file__' in locals() else os.getcwd()
    audio_for_file = dir_path + '\\' + 'audio'

    cols = ['filename']
    if len(results_table) > 0:
        results_table.loc[:, cols] = results_table.loc[:, cols].ffill()
        results_table['Selection'] = results_table.index + 1
        results_table['View'] = 'Spectrogram 1'
        results_table['Channel'] = 1
        results_table['Begin Path'] = audio_for_file + '\\' + results_table.filename
        results_table['File Offset (s)'] = results_table.start
        results_table = results_table.rename(
            columns={"start": "Begin Time (s)", "end": "End Time (s)", "filename": "Begin File"})
        results_table['Low Freq (Hz)'] = 100
        results_table['High Freq (Hz)'] = 1200

        results_table.to_csv(raven_txt, index=False, sep='\t')

    return detections_grp

def run_models(model_folder, audio_folder, output_folder, threshold, max_sd, step_size, batch_size, spectro_file):

    model_names = [model_folder + "\\" + f"rs-model-{i}.kt" for i in range(10)]
    temp_folders = [model_folder + "\\" + f"temp-{i}" for i in range(10)]

    raven_txt = output_folder + '\\' + 'raven-formatted-detections.txt'

    batch_generator = get_batch_generator(spectro_file, audio_folder, step_size, batch_size)
    all_models = load_models(model_names, temp_folders)
    get_detections(batch_generator, all_models, output_folder, threshold, max_sd, raven_txt, audio_folder)

## commandline tool

def command_run_model(args):
    del args['func']
    run_models(**args)

def main():
    import argparse
    parser = argparse.ArgumentParser("Ringed Seal Detector")
    parser.add_argument('model_folder', type=str, help='Folder containing the .kt files')
    parser.add_argument('audio_folder', type=str, help='Folder with audio files')
    parser.add_argument('output_folder', type=str, help='Folder where files will be output into')
    parser.add_argument('--threshold', type=float, help='Threshold at and above to output files', default=0.5)
    parser.add_argument('--max_sd', type=float, help='Maximum allowed standard deviation spread to limit false positives', default=0.15)
    parser.add_argument('--step_size', type=float, help='Step size for processing audio data', default=0.5)
    parser.add_argument('--batch_size', type=float, help='Batch size for processing audio data', default=16)
    parser.add_argument('--spectro_file', type=str, help='Spectro file from the ringed seal documentation',
                        default='spec_config.json')
    parser.set_defaults(func=command_run_model)
    args = parser.parse_args()
    args.func(vars(args))

if __name__ == "__main__":
    main()
