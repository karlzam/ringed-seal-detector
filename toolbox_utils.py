import pandas as pd
import seaborn as sns
pd.options.mode.chained_assignment = None
import sys
import matplotlib.pyplot as plt
from ketos.data_handling import selection_table as sl
from ketos.audio.spectrogram import MagSpectrogram
from ketos.audio.audio_loader import AudioLoader, SelectionTableIterator
import os
import glob

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


def plot_spectrograms(annot_file, data_dir, output_dir, plot_examples, desired_label):
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

    # specify the audio representation
    # TODO: Update to use config file instead
    rep = {'window': 0.05, 'step': 0.001, 'window_func': 'hamming', 'freq_min': 100, 'freq_max': 1200,
           'type': 'MagSpectrogram', 'duration': 5.0}

    # step: 50%: 0.025

    # deal with merging of cells in the annotations table
    for ii in range(0, len(annot)):
        if type(annot.loc[ii][0]) == str:
            filename = annot.loc[ii][0]
        else:
            annot['filename'][ii] = filename

    # standardize tables
    annot_std = sl.standardize(table=annot)

    # create a generator for iterating over all the selections
    generator = SelectionTableIterator(data_dir=data_dir, selection_table=annot_std)

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
                    plt.title(spec.label + 'annot #' + str(ii), y=-0.01)
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

    files = os.listdir(annot)

    for file in files:
        data = pd.read_csv(annot + '\\' + file, delimiter='\t')

        # 'Begin Path' needs to be updated to remove the subfolders and update to new file names
    print('test')





