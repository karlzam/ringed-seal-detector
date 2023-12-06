import pandas as pd

results_table = pd.read_csv(r'C:\Users\kzammit\Documents\Detector\manual_detector_2sec\pp-test\detections_raw.csv')

cols = ['filename']
results_table.loc[:,cols] = results_table.loc[:,cols].ffill()
results_table['Selection'] = results_table.index
results_table['View'] = 'Spectrogram 1'
results_table['Channel'] = 1
results_table['Begin Path'] = results_table.filename
results_table['File Offset (s)'] = results_table.start
results_table = results_table.rename(columns={"start": "Begin Time (s)", "end": "End Time (s)", "filename": "Begin File"})
results_table['Low Freq (Hz)'] = 100
results_table['High Freq (Hz)'] = 1200


results_table.to_excel('raven_formatted_results_pp.xlsx', index=False)






