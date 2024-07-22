import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns

f, axes = plt.subplots(3, 2, sharex=True, figsize=(13, 10))

det_file_ulu = pd.read_excel(r'E:\baseline-with-normalization-reduce-tonal\ulu2023\detections\ensemble\detections-pos-SPL-ulu23.xlsx')
det_file_pp = pd.read_excel(r'E:\baseline-with-normalization-reduce-tonal\pearce-point\detections\ensemble\detections-pos-SPL-pp.xlsx')

det_files = [det_file_pp, det_file_ulu]


for idx, file in enumerate(det_files):

    sub = file[['filename', 'start', 'end', 'Median', 'Dets', 'SPL', 'Manual Class']].copy()

    filled = pd.DataFrame(columns=['bin', 'num-TP', 'num-FP', 'num-FN'])

    TP = sub[sub['Dets'] == 1]
    TP = TP[TP['Manual Class']==1]

    FP = sub[sub['Dets'] == 1]
    FP = FP[FP['Manual Class']==0]

    FN = sub[sub['Dets'] == 0]
    FN = FN[FN['Manual Class']==1]

    # Wanted palette details
    enmax_palette = ["#EFB118", "#4269D0", "#3CA951", "#9498A0", "#A463F2"]
    color_codes_wanted = ['orange', 'blue', 'green', 'grey', 'purple']

    c = lambda x: enmax_palette[color_codes_wanted.index(x)]

    #plt.hist(TP['SPL'], bins=range(80,136))
    sns.histplot(ax=axes[2, idx], data=TP['SPL'], bins=range(80,143), element="step", color = c("orange"), fill=False, linewidth=2)
    sns.histplot(ax=axes[2, idx], data=FP['SPL'], bins=range(80,143), element="step", color= c("blue"), fill=False, linewidth=2)
    sns.histplot(ax=axes[2, idx], data=FN['SPL'], bins=range(80,143), element="step", color= c("purple"), fill=False, linewidth=2)
    sns.histplot(ax=axes[1, idx], data=sub['SPL'], bins=range(80, 141), element="step", color=c("grey"), fill=False, linewidth=2)
    sns.histplot(ax=axes[0, idx], data=sub['SPL'], bins=range(80, 141), element="step", color=c("green"), fill=False,
                 linewidth=2, stat='density')

axes[2, 1].set_xlabel('SPL (dB re 1 uPa)')
axes[2, 0].set_xlabel('SPL (dB re 1 uPa)')
axes[0, 1].set_ylabel('')
axes[0, 0].set_ylabel('Density')
axes[1, 1].set_ylabel('')
axes[2, 1].set_ylabel('')
axes[0, 1].legend(['All' "\n" 'Normalized'], title='Legend', bbox_to_anchor=(1.3, 1.05))
axes[1, 1].legend(['All'], title='Legend', bbox_to_anchor=(1.2, 1.05))
axes[2, 1].legend(['TP', 'FP', 'FN'], title='Legend', bbox_to_anchor=(1.2, 1.05))

axes[0, 0].title.set_text('Pearce Point')
axes[0, 1].title.set_text('Ulukhaktok 2023')

plt.subplots_adjust(wspace=0.11, hspace=0.06)

plt.savefig(r'E:\baseline-with-normalization-reduce-tonal\PP-and-ULU-dets-SPL-w-all-norm.png', bbox_inches='tight', pad_inches=0)



#for ii in range(80, 135):
#    num_TP = len([x for x in TP['SPL'] if ii <= x < ii + 1])
#    num_FP = len([x for x in FP['SPL'] if ii <= x < ii + 1])
##    num_FN = len([x for x in FN['SPL'] if ii <= x < ii + 1])
#    data = pd.DataFrame([{'bin': ii, 'num-TP': num_TP, 'num-FP': num_FP, 'num-FN': num_FN}])
#   filled = pd.concat([filled, data])

#filled.plot(x='bin', y=['num-TP', 'num-FP', 'num-FN'], figsize=(10,5))
#plt.plot(filled['bin'], filled['num-TP'], "#EFB118", label="TP", linestyle='-', markevery=1, marker='o',  markersize=3)
#plt.plot(filled['bin'], filled['num-FP'], "#4269D0", label="FP", linestyle='-', markevery=1, marker='o',  markersize=3)
#plt.plot(filled['bin'], filled['num-FN'], "#3CA951", label="FN", linestyle='-', markevery=1, marker='o',  markersize=3)

#plt.legend(loc="upper right")
#plt.xlabel('SPL (dB re 1 uPa)')
#plt.ylabel('Number of Detections')
#plt.show()

#plt.hist(filled['num-TP'], bins=range(80, 135), color='skyblue', edgecolor='black')

#folder = r'E:\baseline-with-normalization-reduce-tonal\pearce-point\audio' \
#         r'\PAMGuide_Batch_Broadband_Abs_48000ptHannWindow_50pcOlap\*.csv'

#folder = r'D:\ringed-seal-data\Ulu_2023_St5_Site65\test-subset' \
#         r'\PAMGuide_Batch_Broadband_Abs_96000ptHannWindow_50pcOlap\*.csv'

#files = glob.glob(folder)
#filled = pd.DataFrame(columns=["file", "min", "max", "mean", "std"])

#for file in files:
#    temp = pd.read_csv(file)
#    min = temp[temp.columns[1]].min()
#    max = temp[temp.columns[1]].max()
#    mean = temp[temp.columns[1]].mean()
#    std = temp[temp.columns[1]].std()

#    data = pd.DataFrame([{'file': file, 'min': min, 'max': max, 'mean': mean, 'std': std}])

#    filled = pd.concat([filled, data])

#filled.to_excel(r'E:\baseline-with-normalization-reduce-tonal\pearce-point\audio'
#                r'\PAMGuide_Batch_Broadband_Abs_48000ptHannWindow_50pcOlap\stats.xlsx', index=False)

#filled.to_excel(r'D:\ringed-seal-data\Ulu_2023_St5_Site65\test-subset'
#                r'\PAMGuide_Batch_Broadband_Abs_96000ptHannWindow_50pcOlap\stats.xlsx', index=False)
