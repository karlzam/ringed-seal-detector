import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None

def plot_call_length_scatter(annotation_table, output_folder, all_combined):

    df = pd.read_excel(annotation_table)
    output_folder = output_folder

    types = df['label'].unique()

    if all_combined == 0:
        for tdex, typee in enumerate(types):
            df_sub = df.loc[df.label == typee]
            call_length_scat_plot(df_sub, output_folder, typee)
    else:
        call_length_scat_plot(df, output_folder, 'all')


def call_length_scat_plot(df, output_folder, call_type):

    df['delta_time'] = df['end'] - df['start']
    mean = df['delta_time'].mean()
    std = df['delta_time'].std()
    meanpstd = mean + std
    meanmstd = mean - std

    scat_plot = sns.scatterplot(data=df, x=df.index, y="delta_time")
    fig = scat_plot.get_figure()
    ax1 = fig.axes[0]
    ax1.axhline(meanpstd, color='black', linestyle=':')
    ax1.text(-100, mean, "{:.0f}".format(mean), color="red", ha="left", va="center")
    ax1.axhline(meanmstd, color='black', linestyle=':')
    ax1.axhline(mean, color='r', linestyle='dashed')
    plt.title(str(call_type) + ' lengths. mean: ' + str("%.2f" % mean) + '. std: ' +  str("%.2f" % std))
    plt.xlabel('call index')
    plt.ylabel('length (s)')
    plt.savefig(output_folder + r'\\' + 'plot-' + str(call_type) + '.png')
    plt.close()

