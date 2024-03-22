import pandas as pd
import numpy as np
import os
import glob

sel_folder = r'D:\ringed_seal_selection_tables\*.txt'

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

num_div = 1000000000