import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None

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

