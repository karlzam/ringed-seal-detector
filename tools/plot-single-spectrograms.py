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

    for ii in range(0, int(annot)):
        spec = next(loader)
        print('plotting annot #' + str(ii))
        fig = spec.plot(label_in_title=False)
        path = output_dir
        figname = path + "\\" + str(ii) + '.png'
        fig.savefig(figname, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":

    annot_folder = r'E:\baseline-with-normalization-reduce-tonal\spectro\negative-examples'
    data_dir = r"D:\ringed-seal-data"
    spec_file = r'E:\baseline-with-normalization-reduce-tonal\spec_config_100-1200Hz-0.032-hamm-normalized-reduce-tonal.json'

    annot_folders = [annot_folder]

    annot_files = ['neg-examples-sels.xlsx']

    output_dirs = [r'E:\baseline-with-normalization-reduce-tonal\spectro\negative-examples']

    for idx, folder in enumerate(annot_folders):

        # create an audioloader with the spectrograms
        annot, loader = load_audio_seg(annot_folders[idx] + '\\' + annot_files[idx], spec_file, data_dir)

        # take audio segments and create spectrogram representation
        plot_spectrogram(annot, loader, output_dirs[idx])
