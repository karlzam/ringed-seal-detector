import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def by_site_sum(df, sum_output, fig_output):

    # count by site
    site_names = np.unique(df['site_name'])

    df_bySite_sum = pd.DataFrame()
    for site in site_names:
        df_sub = df[df['site_name'] == site]
        df_sub = df_sub[df_sub['call_type'] != 'G']
        num_annots = len(df_sub)
        num_b = sum(df_sub['Barks (KZ)'])
        num_y = sum(df_sub['Yelps (KZ)'])
        num_b_a = len(df_sub[df_sub['call_type'] == 'B'])
        num_by_a = len(df_sub[df_sub['call_type'] == 'BY'])
        print(np.unique(df_sub['call_type']))

        arr = [num_annots, num_b_a, num_by_a, num_b, num_y]

        df_bySite_sum[site] = arr

    df_bySite_sum_T = df_bySite_sum.T

    df_bySite_sum_T = df_bySite_sum_T.rename(columns={0: "total-annot-#", 1: "b-annots", 2: "by-annots",
                                                      3: "total-#-barks", 4: "total-#-yelps"})

    # df_bySite_sum_T.to_excel(sum_output)

    ax = sns.scatterplot(data=df_bySite_sum_T, x=df_bySite_sum_T.index, y="total-annot-#", s=20)
    ax = sns.scatterplot(data=df_bySite_sum_T, x=df_bySite_sum_T.index, y="b-annots", s=11)
    ax = sns.scatterplot(data=df_bySite_sum_T, x=df_bySite_sum_T.index, y="by-annots", s=8)
    ax.legend(labels=["total-annot-#", "b-annots", "by-annots"], loc='upper left')
    plt.xticks(rotation=90)
    plt.title('Annotation Breakdown by Site')
    plt.savefig(fig_output, bbox_inches="tight")

    #df = pd.read_excel(r'C:\Users\kzammit\Documents\Detector\annotation_stats\by_site_sum.xlsx', sheet_name='sum')
    #df=df.drop(axis=0, index=4)

    #ax = sns.scatterplot(data=df, x=df['Unnamed: 0'], y="total-annot-#", s=20)
    #ax = sns.scatterplot(data=df, x=df['Unnamed: 0'], y="b-annots", s=11)
    #ax = sns.scatterplot(data=df, x=df['Unnamed: 0'], y="by-annots", s=8)
    #ax.legend(labels=["total-annot-#", "b-annots", "by-annots"], loc='upper right')
    #plt.title('Annotation Summary by Region')
    #plt.xlabel('Site Name')
    #plt.savefig(r'C:\Users\kzammit\Documents\Detector\annotation_stats\sum2_by_site.png', bbox_inches="tight")

def by_time_sum(df, sumt_output, figt_output):

    df_sub = df[['site_name', 'Real_DateTime_UTC', 'call_type', 'Barks (KZ)', 'Yelps (KZ)']]
    df_sub = df_sub[df_sub['call_type'] != 'G']

    #df_group = df_sub.groupby(pd.Grouper(key='Real_DateTime_UTC', axis=0,
    #                      freq='2D', sort=True)).sum()

    #df_group = df_sub.groupby(pd.Grouper(key='Real_DateTime_UTC', axis=0,

    #                      freq='2D', sort=True))

    """

    df_group = df_sub.groupby(pd.Grouper(key='Real_DateTime_UTC', freq='D')).count()

    df_group = df_group[['site_name']]

    df_group = df_group[df_group['site_name'] != 0]

    df_group = df_group.reset_index()

    df_group['date'] = 'NA'
    for ii in range(0, len(df_group)):
        df_group['date'][ii] = df_group['Real_DateTime_UTC'][ii].strftime('%Y-%m-%d')

    df_group = df_group.set_index('date')

    df_group = df_group.drop(columns=['Real_DateTime_UTC'])
    df_group = df_group.rename(columns={"site_name": "# of annotations"})

    df_group.to_excel(sumt_output)


    plt.figure(figsize=(10, 20))

    ax = df_group.plot(kind='bar')

    for i, t in enumerate(ax.get_xticklabels()):
        if (i % 10) != 0:
            t.set_visible(False)

    ax.get_legend().remove()

    plt.xlabel('Date')
    plt.ylabel('# of Annotations')
    plt.savefig(figt_output, bbox_inches="tight")
    
    """

    df_year = df_sub.groupby(pd.Grouper(key='Real_DateTime_UTC', freq='Y')).count()

    df_year = df_year[['site_name']]

    df_year = df_year.reset_index()

    df_year['date'] = 'NA'
    for ii in range(0, len(df_year)):
        df_year['date'][ii] = df_year['Real_DateTime_UTC'][ii].strftime('%Y-%m-%d')

    df_year = df_year.set_index('date')

    df_year = df_year.drop(columns=['Real_DateTime_UTC'])
    df_year = df_year.rename(columns={"site_name": "# of annotations"})

    df_year.to_excel(r'C:\Users\kzammit\Documents\Detector\annotation_stats\by_year_sum.xlsx')

    plt.figure(figsize=(10, 20))

    ax = df_year.plot(kind='bar')

    ax.get_legend().remove()

    plt.xlabel('Date')
    plt.ylabel('# of Annotations')
    plt.savefig(r'C:\Users\kzammit\Documents\Detector\annotation_stats\sum_by_year.png', bbox_inches="tight")


if __name__ == "__main__":

    df = pd.read_excel(r'C:\Users\kzammit\Documents\Detector\annotation_stats\all_annotations.xlsx')

    #sum_output = r'C:\Users\kzammit\Documents\Detector\annotation_stats\by_site_sum.xlsx'
    #fig_output = r'C:\Users\kzammit\Documents\Detector\annotation_stats\sum_by_site.png'
    #by_site_sum(df, sum_output, fig_output)

    sumt_output = r'C:\Users\kzammit\Documents\Detector\annotation_stats\by_time_sum.xlsx'
    figt_output = r'C:\Users\kzammit\Documents\Detector\annotation_stats\sum_by_time.png'

    by_time_sum(df, sumt_output, figt_output)




