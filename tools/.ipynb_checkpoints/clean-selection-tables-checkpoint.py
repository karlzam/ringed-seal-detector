import pandas as pd
import glob

input_folder = r'D:\ringed_seal_selection_tables'
output_folder = r'D:\ringed_seal_selection_tables\cleaned-annots-for-stats'

files = glob.glob(input_folder + '\*.txt')

for file in files:

    temp = pd.read_csv(file, delimiter="\t")

    print('test')
