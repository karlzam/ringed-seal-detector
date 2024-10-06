import pandas as pd
import seaborn as sns
pd.options.mode.chained_assignment = None
import sys
import matplotlib.pyplot as plt
import matplotlib
from ketos.data_handling import selection_table as sl
from ketos.audio.spectrogram import MagSpectrogram
from ketos.audio.audio_loader import AudioLoader, SelectionTableIterator
import os
import glob
import json

def load_audio_seg(annot_file, spec_file, data_dir):
    """
    Plot spectrograms for review
    :param annot_file: annotation file (xlsx)
    :param spec_file: spectrogram file
    :param data_dir: directory where data is stored (main level)
    :return:
    """

    annot = pd.read_excel(annot_file)

    # something up with loading in the spectro file
    f = open(spec_file)
    spec_info = json.load(f)
    rep = spec_info['spectrogram']

    # deal with merging of cells in the annotations table
    for ii in range(0, len(annot)):
        if type(annot.loc[ii][0]) == str:
            filename = annot.loc[ii][0]
        else:
            filename = annot['filename'][ii]

    # standardize tables
    annot = annot.ffill()
    annot_std = sl.standardize(table=annot)
    print('table standardized? ' + str(sl.is_standardized(annot_std)))

    # create a generator for iterating over all the selections
    generator = SelectionTableIterator(data_dir=data_dir, selection_table=annot_std)

    # Create a loader by passing the generator and the representation to the AudioLoader
    loader = AudioLoader(selection_gen=generator, representation=MagSpectrogram, representation_params=rep, pad=False)

    # print number of segments
    print('Total number of segments is ' + str(loader.num()))
    annots = float(loader.num())

    return annots, loader


def plot_spectrogram(annot, loader, output_dir):

    font = {'family': 'sans',
            'weight': 'normal',
            'size': 20}
    matplotlib.rc('font', **font)

    for ii in range(0, int(annot)):
        spec = next(loader)
        x = spec.get_data()
        extent = (0., spec.duration(), spec.freq_min(), spec.freq_max())  # axes ranges
        plt.imshow(x.T, aspect='auto', origin='lower', extent=extent, vmin=None, vmax=None, cmap='Greys')
        #fig = spec.plot(label_in_title=False)
        path = output_dir
        figname = path + "\\" + str(ii) + '-' + str(loader.selection_gen.sel.Title[ii]) + '.png'
        #plt.title(loader.selection_gen.sel.Title[ii])
        plt.savefig(figname, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":

    annot_folder = r'E:\detector_2\annots\CierrasFiles\selection_tables\annot_format'
    #annot_folder = r'E:\baseline-with-normalization-reduce-tonal\ulu2023\detections\ensemble\plotting'
    data_dir = r"D:\ringed-seal-data"
    spec_file = r'E:\detector_2\spec_config_100-1200Hz-0.032-hamm-normalized.json'

    annot_folders = [annot_folder]

    annot_files = ['ice_bs_cb_sel_table-plot.xlsx']
    #annot_files = ['ulu2023-dets-to-plot-formatted.xlsx']

    output_dirs = [r'E:\detector_2\annots\CierrasFiles\selection_tables\annot_format\spectros']
    #output_dirs = [r'E:\baseline-with-normalization-reduce-tonal\ulu2023\detections\ensemble\plotting']

    for idx, folder in enumerate(annot_folders):

        # create an audioloader with the spectrograms
        annot, loader = load_audio_seg(annot_folders[idx] + '\\' + annot_files[idx], spec_file, data_dir)

        # take audio segments and create spectrogram representation
        plot_spectrogram(annot, loader, output_dirs[idx])
