import pandas as pd
import numpy as np

results_table = pd.read_csv(r'C:\Users\Karlee\Downloads\ULU2022_all_formatted.csv')
audio_folder = r'D:\ringed-seal-data'

cols = ['filename']
results_table.loc[:,cols] = results_table.loc[:,cols].ffill()
results_table['Selection'] = results_table.index +1
results_table['View'] = 'Spectrogram 1'
results_table['Channel'] = 1
#results_table['Begin Path'] = audio_folder + '\\' + results_table.filename
results_table['Begin Path'] = results_table.filename
results_table['File Offset (s)'] = results_table.start
results_table = results_table.rename(columns={"start": "Begin Time (s)", "end": "End Time (s)", "filename": "Begin File"})
results_table['Begin File'] = results_table['Begin File']
#results_table['Low Freq (Hz)'] = 100
#results_table['High Freq (Hz)'] = 1200

results_table.to_csv(r'C:\Users\Karlee\Downloads\ulu-annots-raven.txt',
                     index=False, sep='\t')






