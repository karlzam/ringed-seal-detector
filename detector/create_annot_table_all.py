import sys
import pandas as pd
import glob
import os
import shutil
import random
pd.options.mode.chained_assignment = None


def concat_annot(sel_tables_folder, output_file_name):
    """

    :param sel_tables_folder:
    :param output_file_name:
    :return:
    """

    files = glob.glob(sel_tables_folder + '\*.txt')
    writer = pd.ExcelWriter(output_file_name, engine='xlsxwriter')

    for fdex, file in enumerate(files):

        df = pd.read_csv(file, sep='\t', encoding='latin1')
        df = df.rename(columns={'Keep for detector? Y/X': "keep_drop"})
        df = df[df.keep_drop != 'X']

        if 'CB' in file:
            site_name = file.split('\\')[-1].split('_')[0]
        elif 'Ulukhaktok' in file:
            site_name = file.split('\\')[-1].split('_')[0] + '.' + file.split('\\')[-1].split('_')[1]
        else:
            site_name = file.split('\\')[-1].split('_')[0] + '.' + file.split('\\')[-1].split('_')[1] + '.' + \
                        file.split('\\')[-1].split('_')[2]

        df.insert(0, 'site_name', site_name)

        if fdex == 0:
            df_all = df
        else:
            df_all = df_all.append(df)

    df_all.to_excel(writer, index=False)
    writer.save()
    return (df_all)


def format_annot(df, output_name):
    """

    :param df:
    :param output_name:
    :return:
    """

    writer = pd.ExcelWriter(output_name, engine='xlsxwriter')

    # set start and end time of annotation
    df['start'] = df['File Offset (s)']
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

    df_annot = df_annot.rename(columns={'Call type': "call_type"})
    df_annot.loc[df_annot.call_type.str.contains('bar', na=False), ['label']] = 'bark'
    df_annot.loc[df_annot.call_type.str.contains('Bark', na=False), ['label']] = 'bark'
    df_annot.loc[df_annot.call_type.str.contains('yelp', na=False), ['label']] = 'bark-yelp'
    df_annot.loc[df_annot.call_type.str.contains('growl', na=False), ['label']] = 'bark-yelp-growl'
    df_annot.loc[df_annot.call_type.str.contains('groan', na=False), ['label']] = 'bark-yelp-groan'

    df_annot.to_excel(writer, index=False)
    writer.save()


if __name__ == "__main__":
    # path to folder containing general work, as well as sub-folder with selection tables
    path = r'C:\Users\kzammit\Documents\Work'

    # path to folder with selection tables
    sel_table_path = path + r'\ringed_seal_selection_tables\2022-07-25_All_completed_RS_selection_Tables_by_AR'

    # name of output selection tables excel workbook (just for reference)
    output_file_trim = 'all_annot.xlsx'

    # call the trim_tables function, and set the output excel workbook file name
    all_annot_orig = concat_annot(sel_table_path, output_file_name=output_file_trim)

    format_annot(all_annot_orig, output_name='formatted_annot.xlsx')
