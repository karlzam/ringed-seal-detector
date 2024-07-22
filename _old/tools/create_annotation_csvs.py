import pandas as pd
import glob
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None


def concat_annot(sel_tables_folder):
    """

    :param sel_tables_folder:
    :return:
    """

    files = glob.glob(sel_tables_folder + '\*.txt')

    for fdex, file in enumerate(files):

        df = pd.read_csv(file, sep='\t', encoding='latin1')
        df = df.rename(columns={'KZ Keep? (Y/X/M)': "keep_drop"})
        #df = df[df.keep_drop == 'Y']

        if 'CB' in file:
            site_name = file.split('\\')[-1].split('_')[0]
        elif 'Ulukhaktok' in file:
            site_name = file.split('\\')[-1].split('_')[0] + '.' + file.split('\\')[-1].split('_')[1]
        elif 'ulu.2022' in file:
            site_name = 'ulu.2022'
        elif 'ulu.2023' in file:
            site_name = file.split('\\')[-1].split('_')[0]
        else:
            site_name = file.split('\\')[-1].split('_')[0] + '.' + file.split('\\')[-1].split('_')[1] + '.' + \
                        file.split('\\')[-1].split('_')[2]

        df.insert(0, 'site_name', site_name)

        if fdex == 0:
            df_all = df
        else:
            df_all = pd.concat([df_all, df], ignore_index=True)

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
    df = df.rename(columns={'Begin Path': "filename"})

    # initiate an empty df for the complete annotations
    df_annot = pd.DataFrame(columns=df.columns)

    # get the names of each unique wav file
    unique_files = df['filename'].unique()

    # loop through each unique wav, and set the annot_id for each one
    for fdex, wavF in enumerate(unique_files):

        # create a temp df that only has the entries for this wav file
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
            annot_id += 1

        # append this wav files info to the df_annot df
        df_annot = pd.concat([df_annot, df_temp], ignore_index=True)

    df_annot = df_annot.rename(columns={'Call Type': "call_type"})

    df_annot_sub = df_annot[['site_name', 'Selection', 'filename', 'start', 'end', 'annot_id']]

    df_annot_sub['label'] = df_annot['call_type']

    df_annot_sub.to_excel(writer, index=False)
    writer.close()

    return df_annot_sub


def split_files(df, output_folder):
    """

    :param df:
    :param output_folder:
    :return:
    """

    df = df.rename(columns={'Call type': "label"})

    # drop groans
    df.drop(df[df['label'] == 'G'].index, inplace=True)

    annot_train, annot_val = train_test_split(df, test_size=0.3, random_state=42)

    annot_val, annot_test = train_test_split(annot_val, test_size=0.333, random_state=42)

    # csv versions
    annot_train.to_csv(output_folder + '\\' + 'annotations_train.csv')
    annot_val.to_csv(output_folder + '\\' + 'annotations_val.csv')
    annot_test.to_csv(output_folder + '\\' + 'annotations_test.csv')


if __name__ == "__main__":

    output_dir = r'D:\ringed_seal_selection_tables\ulu2023'

    # path to folder with selection tables
    sel_table_path = r'D:\ringed_seal_selection_tables\ulu2023'

    # name of output selection tables excel workbook (just for reference)
    output_file_trim = output_dir + r'\all_annotations.xlsx'

    # call the trim_tables function, and set the output excel workbook file name
    # note I commented out the intermediate output file
    all_annot_orig = concat_annot(sel_table_path)

    # output an Excel sheet with all the annotations before splitting
    formatted_table = format_annot(all_annot_orig, output_name=output_dir + r'\all_annotations_20240208.xlsx')

    # split files into train and val csvs
    #split_files(formatted_table, output_folder=output_dir)

    #output_dir = r'C:\Users\kzammit\Documents\Detector\manual_dataset\formatted_annots'

    #df = pd.read_excel(r'C:\Users\kzammit\Documents\Detector\manual_dataset\ULU2022_all_annots.xlsx')

    #format_annot(df, output_name=output_dir + r'\ULU2022_all_formatted.xlsx')


