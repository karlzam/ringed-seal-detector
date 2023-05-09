from pretrain_utils import plot_call_length_scatter


# call scatter plot script
annotations_table = r'C:\Users\kzammit\Repos\rs_detector\excel_outputs\formatted_annot_manual.xlsx'
output_fig_folder = r'C:\Users\kzammit\Repos\rs_detector\figs'

plot_call_length_scatter(annotations_table, output_fig_folder, all_combined=0)
