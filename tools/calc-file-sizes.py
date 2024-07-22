import pandas as pd
import numpy as np
import os
import glob

sel_folder = r'D:\ringed_seal_selection_tables\PP\*.txt'

files = glob.glob(sel_folder)

files_unq = []
for file in files:
    temp = pd.read_csv(file,  sep='\t', encoding='latin1')
    unqs = np.unique(temp['Begin Path'])
    size_temp = os.path.getsize(file)
    files_unq.extend(unqs)

size = 0
for file in files_unq:
    size_temp = os.path.getsize(file)
    size += size_temp

file_durations = pd.read_excel(r'C:\Users\kzammit\Repos\ringed-seal-meridian-ketos27\_lockbox\all_file_durations_complete.xlsx')
durs = 0
for unq_file in files_unq:
    file_dur = float(file_durations[file_durations['filename'] == unq_file]['duration'])
    durs += file_dur

print('test')

#file = pd.read_excel(r'D:\ringed_seal_selection_tables\ulu2023\all_positive_annotations_20240304.xlsx')

#unq_files = np.unique(file['filename'])

# size returned in bytes
#size = 0
#for unq_file in unq_files:
#    size_temp = os.path.getsize(unq_file)
#    size += size_temp
#print(size)

#file_durations = pd.read_excel(r'D:\ringed_seal_selection_tables\ulu2023\file_durations_ulu2023.xlsx')

#durs = 0
#for idx, row in file.iterrows():
#    file_dur = float(file_durations[file_durations['filename'] == row['filename']]['duration'])
#    durs += file_dur

print('test')