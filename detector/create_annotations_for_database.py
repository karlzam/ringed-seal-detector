import sys
import openpyxl
import xlsxwriter
import pandas as pd
import glob
import os
import shutil
import random
pd.options.mode.chained_assignment = None


def split_files(df, train_perc, val_perc, output_folder):

    # Calc number of files in the training and validation dataset
    train_num = int(len(df) * train_perc)
    val_num = int(len(df) * val_perc)
    test_num = int(len(df) - train_num - val_num)

    # randomly shuffle the dataframe
    df = df.rename(columns={'Call type': "label"})
    df_shuff = df.sample(frac=1).reset_index(drop=True)

    # separate the training and validation datasets
    # TODO: update so this grabs the first number of them
    df_train = df_shuff.head(train_num)

    writer = pd.ExcelWriter(output_folder + "\\" + 'annotations_train.xlsx', engine='xlsxwriter')
    df_train.to_excel(writer, index=False)
    writer.save()

    # TODO: update so this grabs after the first grab
    df_val = df_shuff.tail(val_num)
    writer = pd.ExcelWriter(output_folder + "\\" + 'annotations_val.xlsx', engine='xlsxwriter')
    df_val.to_excel(writer, index=False)
    writer.save()

    # TODO: add testing command


if __name__ == "__main__":

    all_annot_table = pd.read_excel(r'C:\Users\kzammit\Repos\ringed-seal-meridian\excel_outputs\formatted_annot_manual_mini.xlsx')
    output_folder = r'C:\Users\kzammit\Repos\ringed-seal-meridian\annotation_tables'

    split_files(all_annot_table, train_perc=0.5, val_perc=0.3, output_folder=output_folder)



