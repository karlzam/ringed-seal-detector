import sys
import pandas as pd
import glob
import os
import shutil
import random
pd.options.mode.chained_assignment = None


def split_files(df, split, train_folder, val_folder, all_folder):

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
    train_num = int(len(df) * split)
    val_num = int(len(df) - train_num)

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
    for ii in range(0, len(train_data)):
        if not os.path.isfile(train_folder + "\\" + train_data[ii]):
            shutil.copyfile(all_folder + "\\" + train_data[ii], train_folder + '\\' + train_data[ii])

    val_data = df_val['filename'].tolist()
    for val_data_dex in range(0, len(val_data)):
        if not os.path.isfile(val_folder + "\\" + val_data[val_data_dex]):
            shutil.copyfile(all_folder + "\\" + val_data[val_data_dex], val_folder + '\\' + val_data[val_data_dex])


def sort_files(output_file, sheets_dict, sorted_dir):
    """

    :param file:
    :param sheets_dict:
    :return:
    """

    # Keep track of the number of files copied to ensure the right number have been selected per group
    count = 0

    # dataframe with all selected yes files
    df_all = pd.DataFrame()

    # For each sheet in the Excel workbook
    for fdex in enumerate(sheets_dict):

        # read in the fdex sheet from the workbook
        df = pd.read_excel(output_file, sheet_name=fdex[-1])

        # for now, grabbing the first 3 files in each dataset for "yes" files (arbitrary)
        df1 = df.head(3)
        df_all = df_all.append(df1)

        # for each row of the dataframe, grab the file path from the "begin path" column
        for ii in range(0, len(df1)):
            file = df1.iloc[ii]['begin_path']

            # check if the file exists
            if os.path.isfile(file):

                # create name of sorted folder, which is the dir from the old pathway
                # sorted_folder = sorted_dir + '\\' + file.split("\\")[1]
                sorted_folder = sorted_dir + '\\' + '_all'

                # if folder doesn't exist, make it
                if not os.path.exists(sorted_folder):
                    os.makedirs(sorted_folder)

                # copy the file to the sorted directory (if it's not already there)
                if not os.path.isfile(sorted_folder + '\\' + file.split('\\')[-1]):
                    shutil.copyfile(file, sorted_folder + '\\' + file.split('\\')[-1])
                    print('Copied ' + str(file))

                # increase the copied count by 1
                count = count + 1

            # If the file needing to be copied does not exist, print the name out (shouldn't happen)
            else:
                print('File does not exist:')
                print(str(file))
                sys.exit

        print('Total number of annotation files is ' + str(count))

    writer = pd.ExcelWriter('all-annotations.xlsx', engine='xlsxwriter')
    df_all.to_excel(writer, sheet_name='all', index=False)
    writer.save()

    return df_all


def format_data(file, dict_sheet_names, output_name):
    writer = pd.ExcelWriter(output_name, engine='xlsxwriter')

    for fdex, fname in enumerate(list(dict_sheet_names)):
        df = pd.read_excel(file, sheet_name=fname)

        # set start and end time of annotation
        df['start'] = df['File Offset (s)']

        # if the delta time column exists, calculate the end time
        if 'Delta Time (s)' in df.columns:
            df['end'] = df['File Offset (s)'] + df['Delta Time (s)']

        # if it doesn't calculate the delta time, and then the end time
        else:
            df['Delta Time (s)'] = df['End Time (s)'] - df['Begin Time (s)']
            df['end'] = df['File Offset (s)'] + df['Delta Time (s)']

        # rename the begin path column for easier reference
        df = df.rename(columns={'Begin Path': "begin_path"})

        # initiate an empty df for the complete annotations
        df_annot = pd.DataFrame(columns=df.columns)

        # get the names of each unique wav file
        unique_files = df['begin_path'].unique()

        # loop through each unique wav, and set the annot_id for each one
        for fdex, wavF in enumerate(unique_files):

            # create a temp df that only has the entries for this wav file
            df_temp = df[df.begin_path == wavF]

            # reset the index
            df_temp = df_temp.reset_index(drop=True)

            # set the annot_id column to 'not set' initially
            df_temp['annot_id'] = 'not set'
            df_temp['filename'] = 'not set'

            # start the counter at 0
            annot_id = 0

            # for the number of annotations with this wav file,
            for ii in range(0, len(df_temp)):
                # set the annot_id incrementally
                df_temp['annot_id'][ii] = annot_id
                df_temp['filename'][ii] = wavF.split('\\')[-1]
                annot_id += 1

            # append this wav files info to the df_annot df
            df_annot = df_annot.append(df_temp)

        # print this sites info out to an Excel workbook
        if 'CB300' in fname:
            df_annot.to_excel(writer, sheet_name='CB300', index=False)
        elif 'CB50' in fname:
            df_annot.to_excel(writer, sheet_name='CB50', index=False)
        else:
            df_annot.to_excel(writer, sheet_name=fname, index=False)

    writer.save()


def trim_tables(sel_tables_folder, output_file_name):
    """
    Loop through each selection table and drop the rows corresponding to "X"s in the "keep for detector" column. Create
    a single excel workbook with all trimmed selection tables inside for reference of which were used.
    :param sel_tables_folder: Folder name with selection tables
    :param output_file_name: Name of output excel file
    :return: Sheet names in excel file
    """

    files = glob.glob(sel_tables_folder + '\*.txt')
    writer = pd.ExcelWriter(output_file_name, engine='xlsxwriter')

    for fdex, file in enumerate(files):
        df = pd.read_csv(file, sep='\t', encoding='latin1')
        df = df.rename(columns={'Keep for detector? Y/X': "keep_drop"})
        df = df[df.keep_drop != 'X']

        if 'CB300' in file:
            df.to_excel(writer, sheet_name='CB300', index=False)
        elif 'CB50' in file:
            df.to_excel(writer, sheet_name='CB50', index=False)
        else:
            df.to_excel(writer, sheet_name=file.split('\\')[-1].split('_')[0] + '_' + file.split('\\')[-1].split('_')[1]
                                           + '_' + file.split('\\')[-1].split('_')[2], index=False)
    writer.save()
    return writer.sheets


if __name__ == "__main__":
    # path to folder containing general work, as well as sub-folder with selection tables
    path = r'C:\Users\Karlee\Documents\Masters\Project\Work'

    # path to folder with selection tables
    sel_table_path = path + r'\Data\2022-07-25_All_completed_RS_selection_Tables_by_AR'

    # name of output selection tables excel workbook (just for reference)
    output_file_trim = 'trimmed-tables.xlsx'

    # call the trim_tables function, and set the output excel workbook file name
    writer_sheets = trim_tables(sel_table_path, output_file_name=output_file_trim)

    # set the name of the output formatted data Excel workbook
    output_file_formatted = 'formatted-data.xlsx'

    # calculate the start and end times and number of annotations per file
    format_data(path + '\\' + output_file_trim, writer_sheets, output_file_formatted)

    # name of the directory where sorted files will be stored
    sorted_dir = r'D:\_sorted_files'

    # grab yes files from each directory and copy them into sub folders
    # currently 3 per directory (arbitrary)
    df_all = sort_files(path + "\\" + output_file_formatted, writer_sheets, sorted_dir)

    # split files into test and train data
    split_files(df_all, split=0.6, train_folder=sorted_dir + '\\' + '_train', val_folder=sorted_dir + '\\' + '_val',
                all_folder=sorted_dir + '\\' + '_all')
