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

    # define spectrogram generation parameters
    #spec_par = sl.select(annotations=annot_std, length=1.0, step=1, min_overlap=1, center=False, label=label)

    # create a generator for iterating over all the selections
    generator = SelectionTableIterator(data_dir=data_dir, selection_table=annot_std)

    # Create a loader by passing the generator and the representation to the AudioLoader
    loader = AudioLoader(selection_gen=generator, representation=MagSpectrogram, representation_params=rep, pad=False)

    # print number of segments
    print('Total number of segments is ' + str(loader.num()))
    annots = float(loader.num())

    return annots, loader


def plot_spectrogram(annot, loader, output_dir):
    """
    Plots spectrograms from an audioLoader item, call load_audio_seg first, less options than the old function
    but more useful (?)
    :param annot: annotation table returned from load_audio_seg
    :param loader: audioloader returned from load_audio_seg
    :param output_dir: output path to where you want the image saved
    :return:
    """

    # Ulu good positive numbers: 2, 4, 27, 55


    fig, axs = plt.subplots(2, 2, sharey=True, figsize=(12,9), gridspec_kw={'height_ratios': [1, 1]})

    spec = loader.load(data_dir, loader.selection_gen.get_selection(2)['filename'],
                       loader.selection_gen.get_selection(2)['offset'],
                       loader.selection_gen.get_selection(2)['duration'])
    x = spec.get_data()
    extent = (0., spec.duration(), spec.freq_min(), spec.freq_max())  # axes ranges
    img = axs[0, 0].imshow(x.T, aspect='auto', origin='lower', extent=extent, vmin=None, vmax=None)  # draw image
    axs[0, 0].set_ylabel(spec.freq_ax.label)

    if spec.decibel:
        fig.colorbar(img, ax=axs[0, 0], format='%+2.0f dB')
    else:
        fig.colorbar(img, ax=axs[0, 0], label='Amplitude')

    spec = loader.load(data_dir, loader.selection_gen.get_selection(4)['filename'],
                       loader.selection_gen.get_selection(4)['offset'],
                       loader.selection_gen.get_selection(4)['duration'])
    x = spec.get_data()
    extent = (0., spec.duration(), spec.freq_min(), spec.freq_max())  # axes ranges
    img = axs[0, 1].imshow(x.T, aspect='auto', origin='lower', extent=extent, vmin=None, vmax=None)  # draw image
    #axs[0, 1].set_ylabel(spec.freq_ax.label)

    if spec.decibel:
        fig.colorbar(img, ax=axs[0, 1], format='%+2.0f dB')
    else:
        fig.colorbar(img, ax=axs[0, 1], label='Amplitude')

    spec = loader.load(data_dir, loader.selection_gen.get_selection(27)['filename'],
                       loader.selection_gen.get_selection(27)['offset'],
                       loader.selection_gen.get_selection(27)['duration'])
    x = spec.get_data()
    extent = (0., spec.duration(), spec.freq_min(), spec.freq_max())  # axes ranges
    img = axs[1, 0].imshow(x.T, aspect='auto', origin='lower', extent=extent, vmin=None, vmax=None)  # draw image
    axs[1, 0].set_ylabel(spec.freq_ax.label)
    axs[1, 0].set_xlabel('Time (s)')

    if spec.decibel:
        fig.colorbar(img, ax=axs[1, 0], format='%+2.0f dB')
    else:
        fig.colorbar(img, ax=axs[1, 0], label='Amplitude')

    # ulu 56
    # cb 147
    spec = loader.load(data_dir, loader.selection_gen.get_selection(182)['filename'],
                       loader.selection_gen.get_selection(182)['offset'],
                       loader.selection_gen.get_selection(182)['duration'])
    x = spec.get_data()
    extent = (0., spec.duration(), spec.freq_min(), spec.freq_max())  # axes ranges
    img = axs[1, 1].imshow(x.T, aspect='auto', origin='lower', extent=extent, vmin=None, vmax=None)  # draw image
    #axs[1, 1].set_ylabel(spec.freq_ax.label)
    axs[1, 1].set_xlabel('Time (s)')

    if spec.decibel:
        fig.colorbar(img, ax=axs[1, 1], format='%+2.0f dB')
    else:
        fig.colorbar(img, ax=axs[1, 1], label='Amplitude')

    fig.suptitle('Negative Spectrograms with Normalization')
    #plt.show()
    fig.savefig(r'E:\spectrograms\negatives' + '\\' + 'negative-spectrograms.png')


if __name__ == "__main__":

    annot_file = r'E:\final-baseline-detector\annots\neg\CB-negs-joined.xlsx'
    data_dir = r"D:\ringed-seal-data"
    output_dir = r'E:\spectrograms\negatives'
    spec_file = r'E:\baseline-w-normalization\spec_config_100-1200Hz-0.032-hamm-normalized0.json'

    # create an audioloader with the spectrograms
    annot, loader = load_audio_seg(annot_file, spec_file, data_dir)

    # take audio segments and create spectrogram representation
    plot_spectrogram(annot, loader, output_dir)
