import pandas as pd
import numpy as np
from ketos.neural_networks.dev_utils.detection import filter_by_label, merge_overlapping_detections

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

print('done importing packages')

detections_file = pd.read_csv(r'E:\baseline-with-normalization-reduce-tonal\deploy\ulu2023\deploy-on-audio\detections'
                              r'-model-3.csv')

# Filter the detections for only the positive results
detections_filtered = filter_by_label(detections_file, labels=1).reset_index(drop=True)

detections_grp = merge_overlapping_detections(detections_filtered)

one_min_det = pd.DataFrame(columns=['filename', '0-60s', '60-120s', '120-180s', '180-240s', '240+s'])
all_files = np.unique(detections_file['filename'])
one_min_det['filename'] = all_files
one_min_det.set_index('filename', inplace=True)
one_min_det = one_min_det.fillna(0)

for file in detections_grp['filename'].unique():

    temp = detections_grp[detections_grp['filename']==file]
    for row in temp.iterrows():
        if row[1].end < 60:
            one_min_det.at[file, '0-60s'] = one_min_det.loc[file]['0-60s'] + 1
        elif row[1].start >= 60 and row[1].end < 120:
            one_min_det.at[file, '60-120s'] = one_min_det.loc[file]['60-120s'] + 1
        elif row[1].start >= 120 and row[1].end < 180:
            one_min_det.at[file, '120-180s'] = one_min_det.loc[file]['120-180s'] + 1
        elif row[1].start >= 180 and row[1].end < 240:
            one_min_det.at[file, '180-240s'] = one_min_det.loc[file]['180-240s'] + 1
        elif row[1].start >= 240:
            one_min_det.at[file, '240+s'] = one_min_det.loc[file]['240+s'] + 1

one_min_det['total'] = one_min_det.sum(axis=1)
one_min_det.to_excel(r'E:\baseline-with-normalization-reduce-tonal\deploy\ulu2023\deploy-on-audio\one-min-dets.xlsx')
