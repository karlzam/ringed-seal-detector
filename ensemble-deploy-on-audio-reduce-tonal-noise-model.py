import pandas as pd
from ketos.data_handling.parsing import load_audio_representation
from ketos.neural_networks.resnet import ResNetInterface
from ketos.audio.audio_loader import AudioFrameLoader
from ketos.neural_networks.dev_utils.detection import batch_load_audio_file_data, filter_by_threshold, filter_by_label, merge_overlapping_detections
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

main_folder = r'E:\baseline-with-normalization-reduce-tonal\ulu2023\avg'

model_folder = r'E:\baseline-with-normalization-reduce-tonal\all-models'

model_names = [model_folder + "\\" + "rs-model-0.kt", model_folder + "\\" + "rs-model-1.kt", model_folder + "\\" + "rs-model-2.kt",
            model_folder + "\\" + "rs-model-3.kt", model_folder + "\\" + "rs-model-4.kt", model_folder + "\\" + "rs-model-5.kt",
            model_folder + "\\" + "rs-model-6.kt", model_folder + "\\" + "rs-model-7.kt", model_folder + "\\" + "rs-model-8.kt",
            model_folder + "\\" + "rs-model-9.kt"]

spectro_file = r'E:\baseline-with-normalization-reduce-tonal\spec_config_100-1200Hz-0.032-hamm-normalized-reduce-tonal.json'
output_dir = r'E:\baseline-with-normalization-reduce-tonal\ulu2023\avg'
audio_folder = r'D:\ringed-seal-data\Ulu_2023_St5_Site65\test-subset'
detections_csv = output_dir + '\\' + 'detections-avg.csv'
temp_folder = output_dir + '\\' + 'ringedS_tmp_folder'
pos_detection = output_dir + '\\' + 'grouped-filtered-dets.xlsx'
raven_txt = output_dir + '\\' + 'raven-formatted-detections.txt'

# Step 0.5s each time (overlap of 50% for 1 sec duration)
step_size = 0.5

# Number of samples in batch
batch_size = 16

# Threshold
threshold = 0.5

audio_repr = load_audio_representation(path=spectro_file)

spec_config = audio_repr['spectrogram']

audio_loader = AudioFrameLoader(path=audio_folder, duration=spec_config['duration'],
                                step=step_size, stop=False, representation=spec_config['type'],
                                representation_params=spec_config, pad=False)

batch_generator = batch_load_audio_file_data(loader=audio_loader, batch_size=batch_size)

detections_pos = pd.DataFrame()
detections_neg = pd.DataFrame()

for ibx, batch_data in enumerate(batch_generator):

    for idx, model in enumerate(model_names):

        model_name = model.split('\\')[-1].split('.')[0]

        model = ResNetInterface.load(model_file=model, new_model_folder=temp_folder)

        # Run the model on the spectrogram data from the current batch
        batch_predictions = model.run_on_batch(batch_data['data'], return_raw_output=True)

        if idx == 0:
            # Lets store our data in a dictionary

            raw_output_neg = {'filename': batch_data['filename'], 'start': batch_data['start'],
                              'end': batch_data['end'], 'score-neg': batch_predictions[:, 0]}

            raw_output_pos = {'filename': batch_data['filename'], 'start': batch_data['start'],
                              'end': batch_data['end'], 'score-pos': batch_predictions[:, 1]}

        else:
            raw_output_neg |= {'score-neg-' + str(model_name): batch_predictions[:, 0]}

            raw_output_pos |= {'score-pos-' + str(model_name): batch_predictions[:, 1]}

    detections_pos = pd.concat([detections_pos, pd.DataFrame.from_dict(raw_output_pos)])
    detections_neg = pd.concat([detections_neg, pd.DataFrame.from_dict(raw_output_neg)])

detections_pos.to_excel(output_dir + '\\' + 'detections-pos.xlsx', index=False)
detections_neg.to_excel(output_dir + '\\' + 'detections-neg.xlsx', index=False)

mean_cols_pos = detections_pos.columns[3:]
mean_cols_neg = detections_neg.columns[3:]

detections_pos['mean-pos'] = detections_pos[mean_cols_pos].mean(axis=1)
detections_neg['mean-neg'] = detections_neg[mean_cols_neg].mean(axis=1)

merge_df = detections_pos[['filename', 'start', 'end', 'mean-pos']].copy()
merge_df['mean-neg'] = detections_neg['mean-neg']

scores = []
for row in merge_df.iterrows():
    score = [row[1]['mean-neg'], row[1]['mean-pos']]
    scores.extend([score])

dict = {'filename': merge_df['filename'], 'start': merge_df['start'], 'end': merge_df['end'], 'score': scores}

filter_detections = filter_by_threshold(dict, threshold=threshold)
detections_filtered = filter_by_label(filter_detections, labels=1).reset_index(drop=True)
print(len(detections_filtered))
detections_grp = merge_overlapping_detections(detections_filtered)
print(len(detections_grp))

results_table = pd.read_excel(pos_detection)

cols = ['filename']
results_table.loc[:,cols] = results_table.loc[:,cols].ffill()
results_table['Selection'] = results_table.index +1
results_table['View'] = 'Spectrogram 1'
results_table['Channel'] = 1
results_table['Begin Path'] = audio_folder + '\\' + results_table.filename
results_table['File Offset (s)'] = results_table.start
results_table = results_table.rename(columns={"start": "Begin Time (s)", "end": "End Time (s)", "filename": "Begin File"})
results_table['Begin File'] = results_table['Begin File']
results_table['Low Freq (Hz)'] = 100
results_table['High Freq (Hz)'] = 1200

results_table.to_csv(raven_txt, index=False, sep='\t')