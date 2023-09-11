import pandas as pd
import glob

pd.options.mode.chained_assignment = None


def concat_annot(sel_tables_folder):
    """

    :param sel_tables_folder:
    :return:
    """

    files = glob.glob(sel_tables_folder + '\*.txt')
    # writer = pd.ExcelWriter(output_file_name, engine='xlsxwriter')

    for fdex, file in enumerate(files):

        df = pd.read_csv(file, sep='\t', encoding='latin1')
        df = df.rename(columns={'KZ Keep? (Y/X/M)': "keep_drop"})
        df = df[df.keep_drop == 'Y']

        if 'CB' in file:
            site_name = file.split('\\')[-1].split('_')[0]
        elif 'Ulukhaktok' in file:
            site_name = file.split('\\')[-1].split('_')[0] + '.' + file.split('\\')[-1].split('_')[1]
        elif 'ulu.2022' in file:
            site_name = 'ulu.2022'
        else:
            site_name = file.split('\\')[-1].split('_')[0] + '.' + file.split('\\')[-1].split('_')[1] + '.' + \
                        file.split('\\')[-1].split('_')[2]

        df.insert(0, 'site_name', site_name)

        if fdex == 0:
            df_all = df
        else:
            df_all = df_all.append(df)

    # df_all.to_excel(writer, index=False)
    # writer.save()

    return df_all


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

    # rename the begin-path column for easier reference
    # df = df.rename(columns={'Begin Path': "begin_path"})
    df = df.rename(columns={'Begin Path': "filename"})

    # initiate an empty df for the complete annotations
    df_annot = pd.DataFrame(columns=df.columns)
    # get the names of each unique wav file
    # unique_files = df['begin_path'].unique()
    unique_files = df['filename'].unique()

    # loop through each unique wav, and set the annot_id for each one
    for fdex, wavF in enumerate(unique_files):

        # create a temp df that only has the entries for this wav file
        # df_temp = df[df.begin_path == wavF]
        df_temp = df[df.filename == wavF]

        # reset the index
        df_temp = df_temp.reset_index(drop=True)

        # set the annot_id column to 'not set' initially
        df_temp['annot_id'] = 'not set'

        # start the counter at 0
        annot_id = 0

        # for the number of annotations with this wav file,
        for ii in range(0, len(df_temp)):
            # set the annot_id incrementally
            df_temp['annot_id'][ii] = annot_id
            # df_temp['filename'][ii] = '\\' + wavF.split('D:\\ringed-seal-data\\')[-1]
            # df_temp['filename'][ii] = data_folder + '\\' + df_temp['filename'][ii].split('\\', 1)[1]
            annot_id += 1

        # append this wav files info to the df_annot df
        df_annot = df_annot.append(df_temp)

    df_annot = df_annot.rename(columns={'Call Type (KZ)': "call_type"})

    # df_annot.loc[df_annot.call_type.str.contains('bar', na=False), ['label']] = 'bark'
    # df_annot.loc[df_annot.call_type.str.contains('Bark', na=False), ['label']] = 'bark'
    # df_annot.loc[df_annot.call_type.str.contains('yelp', na=False), ['label']] = 'bark-yelp'
    # df_annot.loc[df_annot.call_type.str.contains('growl', na=False), ['label']] = 'bark-yelp-growl'
    # df_annot.loc[df_annot.call_type.str.contains('groan', na=False), ['label']] = 'bark-yelp-groan'

    df_annot_sub = df_annot[['site_name', 'Selection', 'filename', 'start', 'end', 'annot_id']]

    # df_annot_sub['filename']=df_annot_sub['filename'].split('\\')[-1]

    df_annot_sub['label'] = df_annot['call_type']

    df_annot_sub.to_excel(writer, index=False)
    writer.save()

    return df_annot_sub


def split_files(df, train_perc, val_perc, output_folder):
    """

    :param df:
    :param train_perc:
    :param val_perc:
    :param output_folder:
    :return:
    """

    # TODO: Update this so I can split out test
    # maybe import sklearn train/test split?

    # Calc number of files in the training and validation dataset
    train_num = int(len(df) * train_perc)
    val_num = int(len(df) * val_perc)
    # test_num = int(len(df) - train_num - val_num)

    # randomly shuffle the dataframe
    df = df.rename(columns={'Call type': "label"})
    df_shuff = df.sample(frac=1).reset_index(drop=True)

    # separate the training and validation datasets
    df_train = df_shuff.head(train_num)
    df_val = df_shuff.tail(val_num)

    # xlsx versions
    '''
    writer = pd.ExcelWriter(output_folder + "\\" + 'annotations_train.xlsx', engine='xlsxwriter')
    df_train.to_excel(writer, index=False)
    writer.save()

    writer = pd.ExcelWriter(output_folder + "\\" + 'annotations_val.xlsx', engine='xlsxwriter')
    df_val.to_excel(writer, index=False)
    writer.save()
    '''

    # csv versions
    df_train.to_csv(output_folder + '\\' + 'annotations_train.csv')
    df_val.to_csv(output_folder + '\\' + 'annotations_val.csv')


if __name__ == "__main__":

    output_dir = r'C:\Users\kzammit\Documents\Detector\20230830\inputs'

    # path to folder with selection tables
    sel_table_path = r'C:\Users\kzammit\Documents\Detector\_ringed_seal_selection_tables\finished edits'

    # name of output selection tables excel workbook (just for reference)
    output_file_trim = output_dir + r'\all_annot_20230830.xlsx'

    # call the trim_tables function, and set the output excel workbook file name
    # note I commented out the intermediate output file
    all_annot_orig = concat_annot(sel_table_path)

    # output an Excel sheet with all the annotations before splitting
    formatted_table = format_annot(all_annot_orig, output_name=output_dir + r'\all_annotations_20230830.xlsx')

    # split files into train and val csvs
    split_files(formatted_table, train_perc=0.8, val_perc=0.2, output_folder=output_dir)
