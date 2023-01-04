import sys
import openpyxl
import xlsxwriter
import pandas as pd
import glob
import os
import shutil
import random
pd.options.mode.chained_assignment = None


def split_files(df, train_folder, val_folder, train_perc, val_perc, all_folder):

    # clear files in the training and validation folders
    folder = val_folder
    print('deleting old files from validation folder')
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    folder = train_folder
    print('deleting old files from training folder')
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


    # Calc number of files in the training and validation dataset
    train_num = int(len(df) * train_perc)
    val_num = int(len(df) * val_perc)
    test_num = int(len(df)-train_num-val_num)


    # randomly shuffle the dataframe
    df = df.rename(columns={'Call type': "label"})
    df_shuff = df.sample(frac=1).reset_index(drop=True)

    # separate the training and validation datasets
    df_train = df_shuff.head(train_num)

    writer = pd.ExcelWriter('annotations_train.xlsx', engine='xlsxwriter')
    df_train.to_excel(writer, index=False)
    writer.save()

    df_val = df_shuff.tail(val_num)
    writer = pd.ExcelWriter('annotations_val.xlsx', engine='xlsxwriter')
    df_val.to_excel(writer, index=False)
    writer.save()

    train_data = df_train['filename'].tolist()

    print('test')


    '''
    for ii in range(0, len(train_data)):
        if not os.path.isfile(train_folder + "\\" + train_data[ii]):
            shutil.copyfile(all_folder + "\\" + train_data[ii], train_folder + '\\' + train_data[ii])

    val_data = df_val['filename'].tolist()
    for val_data_dex in range(0, len(val_data)):
        if not os.path.isfile(val_folder + "\\" + val_data[val_data_dex]):
            shutil.copyfile(all_folder + "\\" + val_data[val_data_dex], val_folder + '\\' + val_data[val_data_dex])
    '''

if __name__ == "__main__":
    # path to folder containing general work, as well as sub-folder with selection tables
    path = r'C:\Users\kzammit\Documents\Work\all_data_random_test\database\input\data'
    all_annot_table = pd.read_excel(path + '\\' + 'formatted-data_merged.xlsx')

    split_files(all_annot_table, train_perc=0.6, val_perc=0.31,
                train_folder=path + '\\' + 'train', val_folder=path + '\\' + 'val',
                all_folder=r'E:\all')



