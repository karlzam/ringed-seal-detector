import pandas as pd
import seaborn as sns
pd.options.mode.chained_assignment = None
import sys
import librosa
import matplotlib.pyplot as plt
from ketos.data_handling import selection_table as sl
from ketos.audio.spectrogram import MagSpectrogram
from ketos.audio.audio_loader import AudioLoader, SelectionTableIterator
import os
import glob
import shutil
import json

def calc_file_durations(data_folder):
    """

    :param data_folder:
    :return:
    """

    # calculate the file durations
    # file_durations_val = sl.file_duration_table(data_folder)
    # file_durations_train = sl.file_duration_table(data_folder)

    folders = [f.path for f in os.scandir(data_folder) if f.is_dir()]

    # temp fix to get around corrupt ULU files
    #folders = folders[0:6]

    # moving onto corrupt ULU files
    #folders = folders[6:]

    # for new ulu 2022 files
    folders = [folders[-1]]

    file_durations = pd.DataFrame()
    for folder in folders:

        folder_name = folder.split('\\')[-1]

        # get the durations for all the files within this folder
        file_durations_for_folder = sl.file_duration_table(data_folder + "\\" + str(folder_name))

        # it only keeps the wav filename, need to append the path back onto it I think? Let's try that
        file_durations_for_folder['fixed_filename'] = folder + "\\" + file_durations_for_folder['filename']

        # drop the og filename column
        file_durations_for_folder = file_durations_for_folder.drop(['filename'], axis=1)

        # rename the appended filename
        file_durations_for_folder = file_durations_for_folder.rename(columns={'fixed_filename': 'filename'})

        #file_durations = file_durations.append(sl.file_duration_table(data_folder + "\\" + str(folder_name)))
        file_durations = file_durations.append(file_durations_for_folder)

    #print(len(file_durations))
    #file_durations.to_excel('Ulu_durations.xlsx', index=False)

    #file_durations.to_excel('file_durations_minusUlu.xlsx', index=False)

    file_durations.to_excel('file_durations_ulu2022.xlsx', index=False)


def plot_call_length_scatter(annotation_table, output_folder, all_combined):
    """
    Plot a scatter plot for call lengths, calling the call_length_scat_plot function
    :param annotation_table: table with annotations
    :param output_folder: folder where you'd like the plots output to
    :param all_combined: 1 for all call types plotted together, 0 for a plot per call type
    :return:
    """

    df = pd.read_excel(annotation_table)
    output_folder = output_folder

    # get the number of different types of calls
    types = df['label'].unique()

    # if you don't want one plot for everything
    if all_combined == 0:
        # loop through each type and plot a scatter plot
        for tdex, typee in enumerate(types):
            # make a sub dataframe with just that call type
            df_sub = df.loc[df.label == typee]
            # call plotting function
            call_length_scat_plot(df_sub, output_folder, typee)
    # or else plot one plot for all call types together
    else:
        # call plotting function
        call_length_scat_plot(df, output_folder, 'all')


def call_length_scat_plot(df, output_folder, call_type):
    """
    Plot call lengths as a scatter plot
    :param df: annotation table
    :param output_folder: where you want the output to be
    :param call_type: name for the plot (ie what type of call it is)
    :return:
    """

    # calculate the length of time of each call
    df['delta_time'] = df['end'] - df['start']
    # calculate the mean
    mean = df['delta_time'].mean()
    # calculate the standard deviation
    std = df['delta_time'].std()

    # plot a seaborn scatterplot of the index vs the delta time
    scat_plot = sns.scatterplot(data=df, x=df.index, y="delta_time")
    fig = scat_plot.get_figure()
    ax1 = fig.axes[0]
    # plot horizontal lines for the mean and the mean +- std
    ax1.axhline(mean + std, color='black', linestyle=':')
    ax1.text(-100, mean, "{:.0f}".format(mean), color="red", ha="left", va="center")
    ax1.axhline(mean - std, color='black', linestyle=':')
    ax1.axhline(mean, color='r', linestyle='dashed')
    # set the title to include the mean and std
    plt.title(str(call_type) + ' lengths. mean: ' + str("%.2f" % mean) + '. std: ' +  str("%.2f" % std))
    plt.xlabel('call index')
    plt.ylabel('length (s)')
    # save the figure
    plt.savefig(output_folder + r'\\' + 'plot-' + str(call_type) + '.png')
    plt.close()


def plot_spectrograms(annot_file, spec_file, data_dir, output_dir, plot_examples, desired_label):
    """
    Plot spectrograms for review
    :param annot_file: annotation file (xlsx)
    :param data_dir: directory where data is stored (main level)
    :param output_dir: folder to output spectro plots
    :param plot_examples: "1" if want to plot first 30 examples of desired label, "0" if want to plot ALL
    :param desired_label: ex. "1" for barks, "0" for noise
    :return:
    """

    annot = pd.read_excel(annot_file)

    # something up with loading in the spectro file
    #f = open(spec_file)
    #spec_info = json.load(f)
    #rep = spec_info['spectrogram']

    # specify the audio representation
    rep = {'window': 0.05, 'step': 0.001, 'window_func': 'hamming', 'freq_min': 100, 'freq_max': 1200,
           'type': 'MagSpectrogram', 'duration': 5.0}

    # step: 50%: 0.025

    # deal with merging of cells in the annotations table
    for ii in range(0, len(annot)):
        if type(annot.loc[ii][0]) == str:
            filename = annot.loc[ii][0]
        else:
            filename = annot['filename'][ii]

    # standardize tables
    annot_std = sl.standardize(table=annot)
    print('table standardized? ' + str(sl.is_standardized(annot_std)))

    spec_eq_length = sl.select(annotations=annot_std, length=2.0, step=1, min_overlap=0.8, center=False)

    # create a generator for iterating over all the selections
    #generator = SelectionTableIterator(data_dir=data_dir, selection_table=annot_std)
    generator = SelectionTableIterator(data_dir=data_dir, selection_table=spec_eq_length)

    # Create a loader by passing the generator and the representation to the AudioLoader
    loader = AudioLoader(selection_gen=generator, representation=MagSpectrogram, representation_params=rep)

    # print number of segments
    print(loader.num())
    annots = float(loader.num())
    # load and plot the first selection

    if plot_examples == 0:
        for ii in range(0, int(annots)):
            spec = next(loader)
            if int(spec.label) == desired_label:
                print('plotting annot #' + str(ii))
                fig = spec.plot()
                path = output_dir
                figname = path + "\\" + str(ii) + '.png'
                plt.title(str(spec.label) + ', annot #' + str(ii), y=-0.01)
                fig.savefig(figname)
                plt.close(fig)

    if plot_examples == 1:
        examples = 0
        for ii in range(0, int(annots)):
            spec = next(loader)
            if int(spec.label) == desired_label:
                if examples < 30:
                    print('plotting annot #' + str(ii))
                    fig = spec.plot()
                    path = output_dir
                    figname = path + "\\" + str(ii) + '.png'
                    plt.title(str(spec.label) + 'annot #' + str(ii), y=-0.01)
                    fig.savefig(figname)
                    plt.close(fig)
                    examples += 1
                if examples == 30:
                    sys.exit()


def write_file_locations(input_folder, output_file_name):
    """
    Write Excel file containing all file locations from data dir for future reference
    :param input_folder:
    :param output_file_name:
    :return:
    """
    writer = pd.ExcelWriter(output_file_name, engine='xlsxwriter')
    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]

    all_files = []
    for folder in subfolders:
        site_name = folder.split('\\')[-1]
        files = glob.glob(folder + '\*.wav')
        for file in files:
            filename = file.split('\\')[-1]
            df_row = [site_name, filename]
            all_files.append(df_row)

    df_all = pd.DataFrame(all_files, columns=['folder', 'filename'])
    df_all.to_excel(writer, index=False)
    writer.save()

def rename_ulu_2022_files(data_folder, annot):

    ### rename and move files to not be in subfolders
    '''
    # get names of subfolders
    subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]

    for folder in subfolders:

        sub_sub_folders = [f.path for f in os.scandir(folder) if f.is_dir()]

        for site_path in sub_sub_folders:

            site = site_path.split('\\')[-1]

            station = site_path.split('\\')[3]

            file_begin = station + '_' + site + '_'

            entries = os.listdir(site_path)

            for entry in entries:
                old_name = site_path + '\\' + entry
                new_name = data_folder + '\\' + file_begin + entry
                os.rename(old_name, new_name)

    '''

    ### update annotation tables
    # example name: ST1_Site2_67653638.220513120735.wav
    # column name: Begin Path
    # column entry: D:\ringed-seal-data\Ulu_2022\ST2\Site5\67145740.220513172342.wav

    files = os.listdir(annot)

    for file in files:
        data = pd.read_csv(annot + '\\' + file, delimiter='\t')
        data_new = pd.DataFrame()
        for idx, row in data.iterrows():

            # When I updated begin path
            #new_name = row['Begin Path'].split('\\')[0] + '\\' + row['Begin Path'].split('\\')[1] + '\\' + \
            #           row['Begin Path'].split('\\')[2] + '\\' + row['Begin Path'].split('\\')[3] + '_' + \
            #           row['Begin Path'].split('\\')[4] + '_' + row['Begin Path'].split('\\')[5]

            #row['Begin Path'] = new_name

            # updating begin file

            #new_name_2 = row['Begin Path'].split('\\')[-1].split('_')[0] + '_' + \
            #             row['Begin Path'].split('\\')[-1].split('_')[1] + '_' + row['Begin File']

            #row['Begin File'] = new_name_2


            row['Begin File'] = row['Begin Path'].split("\\")[-1]

            data_new[idx] = row

        data_new = data_new.swapaxes("index", "columns")

        data_new.to_csv(path_or_buf = annot + '\\' + file.split('.txt')[0] + '_2.txt', sep='\t')

        #... the begin file column also needs to be updated. breaking the annotation step.


def inspect_audio_files(wav):
    """

    :param wav:
    :return:
    """

    sig, rate = librosa.load(wav, sr=None)

    return sig, rate

def copy_audio_files(val_csv, audio_folder):
    """

    :param val_csv:
    :param audio_folder:
    :return:
    """

    validation_files = pd.read_csv(val_csv)

    for idex, row in validation_files.iterrows():
        shutil.copyfile(validation_files.loc[idex]['filename'], audio_folder + '\\' +
                        validation_files.loc[idex]['filename'].split('\\')[-1])









