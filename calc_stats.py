import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.transforms as transforms

df = pd.read_excel(r'C:\Users\kzammit\Repos\rs_detector\excel_outputs\formatted_annot_manual.xlsx')

types = df['label'].unique()

for tdex, typee in enumerate(types):
    df_sub = df.loc[df.label==typee]
    type_name = typee
    df_sub['delta_time'] = df_sub['end'] - df_sub['start']

    mean = df_sub['delta_time'].mean()
    std = df_sub['delta_time'].std()
    meanpstd = mean + std
    meanmstd = mean - std



    scat_plot = sns.scatterplot(data=df_sub, x=df_sub.index, y="delta_time")
    fig = scat_plot.get_figure()
    ax1 = fig.axes[0]
    ax1.axhline(meanpstd, color='black', linestyle=':')
    ax1.text(-100, mean, "{:.0f}".format(mean), color="red", ha="left", va="center")
    ax1.axhline(meanmstd, color='black', linestyle=':')
    ax1.axhline(mean, color='r', linestyle='dashed')
    #ax1.xlim(0, max(df_sub.index))
    #ax1.ylim(meanmstd, max(df_sub['delta_time'])+2)
    plt.title(str(typee) + ' lengths')
    plt.xlabel('call index')
    plt.ylabel('length (s)')
    plt.savefig('plot-' + str(typee) + '.png')
    plt.close()

'''
    plt.scatter(df_sub.index, df_sub['delta_time'], color='blue')
    plt.axhline(y=meanpstd, color='black', linestyle=':')
    plt.axhline(y=meanmstd, color='black', linestyle=':')
    plt.axhline(y=mean, color='r', linestyle='dashed')
    plt.xlim(0, max(df_sub.index))
    plt.ylim(meanmstd, max(df_sub['delta_time'])+2)
    plt.title(str(typee) + ' lengths')
    plt.xlabel('call index')
    plt.ylabel('length (s)')
    plt.savefig('plot' + str(typee) + '.png')
    plt.close()

'''