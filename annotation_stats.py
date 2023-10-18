import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import glob


if __name__ == "__main__":

    path = r'D:\ringed_seal_selection_tables\*.txt'
    files = glob.glob(path)

    for file in files:

        f = pd.read_csv(file,  delimiter='\t')

        for index, row in f.iterrows():

            if 'CB300' in file:

                name = row['Begin File']

                print('test')

