import pandas
import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import glob
pd.options.mode.chained_assignment = None

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

def by_time_sum():

    df = pd.read_excel(r'C:\Users\kzammit\Documents\Detector\annotation_stats\all_annotations.xlsx')

    df_sub = df[['site_name', 'Real_DateTime_UTC', 'call_type', 'Barks (KZ)', 'Yelps (KZ)']]
    df_sub = df_sub[df_sub['call_type'] != 'G']

    main_folder = r'C:\Users\kzammit\Documents\Detector\annotation_stats'

    #freqs = ['D', 'Y', 'M']
    freqs = ['M']

    #excel_names = [main_folder + '\\' + 'by_day_sum.xlsx', main_folder + '\\' + 'by_year_sum.xlsx',
    #               main_folder + '\\' + 'by_month_sum.xlsx']

    excel_names = [main_folder + '\\' + 'by_month_sum.xlsx']

    #fig_names = [main_folder + '\\' + 'day_sum.png', main_folder + '\\' + 'year_sum.png',
    #             main_folder + '\\' + 'month_sum.png']

    fig_names = [main_folder + '\\' + 'month_sum.png']

    label_freq = [1]


    for ii in range(0, len(freqs)):

        df_group = df_sub.groupby(pd.Grouper(key='Real_DateTime_UTC', freq=freqs[ii])).count()

        df_group = df_group[['site_name']]

        df_group = df_group[df_group['site_name'] != 0]

        df_group = df_group.reset_index()

        df_group['date'] = 'NA'
        for jj in range(0, len(df_group)):
            df_group['date'][jj] = df_group['Real_DateTime_UTC'][jj].strftime('%Y-%m-%d')

        df_group = df_group.set_index('date')

        df_group = df_group.drop(columns=['Real_DateTime_UTC'])
        df_group = df_group.rename(columns={"site_name": "# of annotations"})

        df_group.to_excel(excel_names[ii])

        plt.figure(figsize=(10, 20))

        ax = df_group.plot(kind='bar')

        for i, t in enumerate(ax.get_xticklabels()):
            if (i % label_freq[ii]) != 0:
                t.set_visible(False)

        ax.get_legend().remove()

        plt.xlabel('Date')
        plt.ylabel('# of Annotations')
        plt.savefig(fig_names[ii], bbox_inches='tight')


def monthly_site_plots():

    df = pd.read_excel(r'C:\Users\kzammit\Documents\Detector\annotation_stats\manual_monthly.xlsx')

    # sns.histplot(x=df['month'], hue=df['site_name_comb'], palette=['lightgreen', 'pink', 'orange', 'skyblue'])

    CB_df = df[df['site_name_comb'] == 'CB']
    KK_df = df[df['site_name_comb'] == 'KK']
    #ULU2022_df = df[df['site_name'] == 'ulu.2022']
    #ULU_df = df[df['site_name'] == 'Ulukhaktok.2017-2018']
    ULU_df = df[df['site_name_comb'] == 'ULU']
    PP_df = df[df['site_name_comb'] == 'PP']

    main_folder = r"C:\Users\kzammit\Documents\Detector\annotation_stats\plots"
    sites = [CB_df, KK_df, ULU_df, PP_df]
    titles = ['CB', 'KK', 'ULU', 'PP']
    name = ['CB_by_month.png', 'KK_by_month.png', 'ULU_by_month.png', 'PP_by_month.png']

    hue_order = ['BY', 'B']
    sns.set_palette("Set1")

    for jj in range(0, len(titles)):
        # hist_plot = sns.histplot(x=sites[jj]['month'], hue=sites[jj]['call_type'],
        #                         hue_order=hue_order, alpha=0.5, element='step').set_title(titles[jj])
        # fig = hist_plot.get_figure()

        ax = sns.countplot(x=sites[jj]['month'], hue=sites[jj]['call_type'],
                           hue_order=hue_order, alpha=1)

        for ii in ax.containers:
            ax.bar_label(ii, )

        ax.set_title(titles[jj])
        fig = ax.get_figure()
        fig.savefig(main_folder + '\\count_' + name[jj])
        fig.clf()

def find_times(df):

    df_edit = df[df['site_name'] != 'ulu.2022']

    ULU2022_df = df[df['site_name'] == 'ulu.2022']
    ULU2022_df.sort_values(by='Real_DateTime_UTC', inplace=True)

    CB_df = df_edit[df_edit['site_name_comb'] == 'CB']
    CB_df.sort_values(by='Real_DateTime_UTC', inplace=True)

    KK_df = df_edit[df_edit['site_name_comb'] == 'KK']
    KK_df.sort_values(by='Real_DateTime_UTC', inplace=True)

    ULU_df = df_edit[df_edit['site_name_comb'] == 'ULU']
    ULU_df.sort_values(by='Real_DateTime_UTC', inplace=True)

    PP_df = df_edit[df_edit['site_name_comb'] == 'PP']
    PP_df.sort_values(by='Real_DateTime_UTC', inplace=True)

    main_folder = r'C:\Users\kzammit\Documents\Detector\manual_dataset'

    CB_df.to_excel(main_folder + '\\' + 'CB_all_annots.xlsx', index=False)
    PP_df.to_excel(main_folder + '\\' + 'PP_all_annots.xlsx', index=False)
    KK_df.to_excel(main_folder + '\\' + 'KK_all_annots.xlsx', index=False)
    ULU_df.to_excel(main_folder + '\\' + 'ULU_all_annots.xlsx', index=False)
    ULU2022_df.to_excel(main_folder + '\\' + 'ULU2022_all_annots.xlsx', index=False)

    CB_vals = [130, 37, 18]

    CB_train = CB_df[0:CB_vals[0]]
    CB_train.to_excel(main_folder + '\\' + 'CB_train_annots.xlsx', index=False)

    CB_val = CB_df[CB_vals[0]:CB_vals[0] + CB_vals[1]]
    CB_val.to_excel(main_folder + '\\' + 'CB_val_annots.xlsx', index=False)

    CB_test = CB_df.tail(CB_vals[2])
    CB_test.to_excel(main_folder + '\\' + 'CB_test_annots.xlsx', index=False)

    KK_vals = [1230, 348, 171]

    KK_train = KK_df[0:KK_vals[0]]
    KK_train.to_excel(main_folder + '\\' + 'KK_train_annots.xlsx', index=False)

    KK_val = KK_df[KK_vals[0]:KK_vals[0] + KK_vals[1]]
    KK_val.to_excel(main_folder + '\\' + 'KK_val_annots.xlsx', index=False)

    KK_test = KK_df.tail(KK_vals[2])
    KK_test.to_excel(main_folder + '\\' + 'KK_test_annots.xlsx', index=False)

    ULU_vals = [634, 179, 88]

    ULU_train = ULU_df[0:ULU_vals[0]]
    ULU_train.to_excel(main_folder + '\\' + 'ULU_train_annots.xlsx', index=False)

    ULU_val = ULU_df[ULU_vals[0]:ULU_vals[0] + ULU_vals[1]]
    ULU_val.to_excel(main_folder + '\\' + 'ULU_val_annots.xlsx', index=False)

    ULU_test = ULU_df.tail(ULU_vals[2])
    ULU_test.to_excel(main_folder + '\\' + 'ULU_test_annots.xlsx', index=False)

    ULU2022_vals = [949, 274, 139]

    ULU2022_train = ULU2022_df[0:ULU2022_vals[0]]
    ULU2022_train.to_excel(main_folder + '\\' + 'ULU2022_train_annots.xlsx', index=False)

    ULU2022_val = ULU2022_df[ULU2022_vals[0]:ULU2022_vals[0] + ULU2022_vals[1]]
    ULU2022_val.to_excel(main_folder + '\\' + 'ULU2022_val_annots.xlsx', index=False)

    ULU2022_test = ULU2022_df.tail(ULU2022_vals[2])
    ULU2022_test.to_excel(main_folder + '\\' + 'ULU2022_test_annots.xlsx', index=False)

    print('test')


def fancy_plot():

    df = pd.read_excel(r'C:\Users\kzammit\Documents\Detector\manual_dataset\manual_monthly_splitUlu.xlsx')

    #sites = ['CB', 'ULU', 'KK']
    main_folder = r'C:\Users\kzammit\Documents\Detector\manual_dataset'
    #names = ['CB_split.png', 'ULU_split.png', 'KK_split.png']
    #titles = ['CB', 'ULU', 'KK']

    sites = ['CB']
    names = ['CB_split.png']
    titles = ['CB']

    for ii in range(0, len(sites)):

        df2 = df[df['site_name_comb'] == sites[ii]]

        df_monthly = df2.groupby(pd.Grouper(key='Real_DateTime_UTC', freq='D')).count()
        df_monthly = df_monthly.reset_index()

        df_monthly['date'] = 'NA'
        for jj in range(0, len(df_monthly)):
            #df_monthly['date'][jj] = df_monthly['Real_DateTime_UTC'][jj].strftime('%Y-%m-%d')
            df_monthly['date'][jj] = df_monthly['Real_DateTime_UTC'][jj].to_datetime64()

        plt.figure(figsize=(10, 10))

        #ax = sns.histplot(df_monthly['site_name'], x=df_monthly['date'])

        df_monthly = df_monthly.rename(columns={"site_name": "count"})

        # This one works currently with the vertical line plot, not sure why the histo isn't
        #ax = sns.scatterplot(df_monthly, x='date', y='site_name')

        ax = sns.barplot(df_monthly, x='date', y='count')

        ax.axvline(df_monthly['date'][168], label='test', color='r')

        # This works for scatter
        # ax.axvline(df_monthly['Real_DateTime_UTC'][167].strftime('%Y-%m-%d'), label='test', color='r')

        for i, t in enumerate(ax.get_xticklabels()):
            if (i % 10) != 0:
                t.set_visible(False)

        plt.xticks(rotation=90)

        plt.title(titles[ii])
        plt.xlabel('Date')
        plt.ylabel('# of Annotations')
        plt.legend()
        plt.savefig(main_folder + '\\' + names[ii], bbox_inches='tight')

    print('test')



    print('test')

if __name__ == "__main__":

    '''
    # Code used to do the manual monthly excel file 
    df_sub = df[['site_name', 'Real_DateTime_UTC', 'call_type']]
    df_sub = df_sub[df_sub['call_type'] != 'G']

    sort = list(df_sub.groupby(df_sub['Real_DateTime_UTC'].dt.month))
    '''

    df = pd.read_excel(r'C:\Users\kzammit\Documents\Detector\manual_dataset\all_annotations_add_combname_manual.xlsx')
    find_times(df)

    #monthly_site_plots()

    #fancy_plot()





