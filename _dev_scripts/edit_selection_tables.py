import os
import pandas as pd

def drop_out_of_bounds_sel(sel_table, name, file_durations):

    sel_table = sel_table.ffill()

    print('The length of df before dropping is ' + str(len(sel_table)))
    # add step here to drop selections past the end of the file
    # train

    drop_rows_after = []

    for idex, row in sel_table.iterrows():

        # filename is row[0], end time is idex.end
        index = file_durations.loc[file_durations['filename'] == row.filename].index[0]
        duration = file_durations['duration'][index]

        if duration < row.end:
            # drop the row corresponding to that sel_id and filename from the dataframe
            drop_rows_after.append(idex)

        if row.start < 0:
            drop_rows_after.append(idex)

    print('The number of rows to drop is ' + str(len(drop_rows_after)))
    sel_table = sel_table.drop(drop_rows_after)

    print('The new length of sel table is ' + str(len(sel_table)))

    sel_table.to_excel(name)


if __name__ == "__main__":

    #neg_folder = r'C:\Users\kzammit\Documents\Detector\manual_detector_2sec\inputs\annotations\original_sels\negatives'
    #edit_neg_folder = r'C:\Users\kzammit\Documents\Detector\manual_detector_2sec\inputs\annotations\edited_sels\negatives'

    pos_folder = r'C:\Users\kzammit\Documents\Detector\manual_detector_2sec\inputs\annotations\original_sels\positives'
    edit_pos_folder = r'C:\Users\kzammit\Documents\Detector\manual_detector_2sec\inputs\annotations\edited_sels\positives'

    #files_neg = [x for x in os.listdir(neg_folder) if x.endswith('.xlsx')]

    files_pos = [x for x in os.listdir(pos_folder) if x.endswith('.xlsx')]

    file_durations = pd.read_excel(r'C:\Users\kzammit\Repos\ringed-seal-meridian-ketos27\_lockbox'
                                   r'\all_file_durations_complete.xlsx')

    #for file in files_neg:
    #    name = edit_neg_folder + '\\' + str(file)
    #    drop_out_of_bounds_sel(pd.read_excel(neg_folder + '\\' + str(file)), name, file_durations)

    for file in files_pos:
        name = edit_pos_folder + '\\' + str(file)
        print(name)
        drop_out_of_bounds_sel(pd.read_excel(pos_folder + '\\' + str(file)), name, file_durations)



