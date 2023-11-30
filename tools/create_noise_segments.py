import pandas as pd
import json
import matplotlib.pyplot as plt
from ketos.data_handling import selection_table as sl
from ketos.audio.spectrogram import MagSpectrogram
from ketos.audio.audio_loader import AudioLoader, SelectionTableIterator


def create_tables(annot_tables, main_folder, neg_names, file_durations, output_dirs, spec_file, data_dir, num_annots):

    for idx, table in enumerate(annot_tables):

        #annot = pd.read_csv(main_folder + '\\' + table)
        annot = pd.read_excel(main_folder + '\\' + table)
        annot['label'] = 'nan'

        #std_annot = sl.standardize(table=annot, labels=["B", "BY"], start_labels_at_1=True, trim_table=True)
        std_annot = sl.standardize(table=annot, labels=["nan"], start_labels_at_1=False, trim_table=True)
        #std_annot['label'] = std_annot['label'].replace(2, 1)

        #positives = sl.select(annotations=std_annot, length=2.0, center=True, label=1)

        file_durations2 = file_durations[file_durations['filename'].isin(annot['filename'])]

        #negatives = sl.create_rndm_selections(annotations=std_annot, files=file_durations2,
        #                                            length=2.0, num=len(num_annots[idx]), trim_table=True)

        negatives = sl.create_rndm_selections(annotations=std_annot, files=file_durations2,
                                                length=2.0, num=num_annots[idx], trim_table=True, no_overlap=True,
                                                buffer=4)

        negatives = negatives.reset_index()

        negatives.to_excel(main_folder + '\\' + neg_names[idx], index=False)

        #positives = positives.reset_index()
        #positives.to_excel(main_folder + '\\' + pos_names[idx], index=False)

        # annot_std = sl.standardize(table=negatives)
        annot_std = sl.standardize(table=negatives, start_labels_at_1=False)
        print('table standardized? ' + str(sl.is_standardized(annot_std)))

        annot_std.to_excel(main_folder + '\\std_' + neg_names[idx])

        f = open(spec_file)
        spec_info = json.load(f)
        rep = spec_info['spectrogram']

        generator = SelectionTableIterator(data_dir=data_dir, selection_table=annot_std)

        loader = AudioLoader(selection_gen=generator, representation=MagSpectrogram, representation_params=rep,
                             pad=False)

        annot_num = float(loader.num())

        for ii in range(0, int(annot_num)):
            spec = next(loader)
            print('plotting annot #' + str(ii))
            #plt.use('Agg')
            fig = spec.plot()
            path = output_dirs[idx]
            figname = path + "\\" + str(ii) + '.png'
            # plt.title(str(spec.label) + ', annot #' + str(ii), y=-0.01)
            fig.savefig(figname, bbox_inches='tight')
            plt.close()
            #plt.close(fig)


if __name__ == "__main__":


    # main_folder = r'C:\Users\kzammit\Documents\Detector\manual_dataset\formatted_annots'

    #annot_tables = ['CB_all_formatted.csv', 'ULU_all_formatted.csv', 'ULU2022_all_formatted.csv',
    #                'KK_all_formatted.csv', 'PP_all_formatted.csv']

    #neg_names = ['CB_negatives.xlsx', 'ULU_negatives.xlsx', 'ULU2022_negatives.xlsx',
    #             'KK_negatives.xlsx', 'PP_negatives.xlsx']

    #pos_names = ['CB_positives.xlsx', 'ULU_positives.xlsx', 'ULU2022_positives.xlsx',
    #             'KK_positives.xlsx', 'PP_positives.xlsx']

    #file_durations = pd.read_excel(r'C:\Users\kzammit\Documents\Detector\manual_dataset\formatted_annots'
    #                               r'\all_file_durations_complete.xlsx')

    # output_dir_main = r'C:\Users\kzammit\Documents\Detector\manual_dataset\negatives'


    main_folder = r'C:\Users\kzammit\Documents\Detector\manual_dataset\with_discards'

    output_dir_main = r'C:\Users\kzammit\Documents\Detector\manual_dataset\negatives'

    #annot_tables = ['CB_all_annots_with_discards.xlsx', 'ULU_all_annots_with_discards.xlsx',
    #                'ULU2022_all_annots_with_discards.xlsx', 'KK_all_annots_with_discards.xlsx',
    #                'PP_all_annots_with_discards.xlsx']

    annot_tables = ['PP_all_annots_with_discards.xlsx']

    #output_dirs = [output_dir_main + '\\' + 'CB', output_dir_main + '\\' + 'ULU', output_dir_main + '\\' + 'ULU2022',
    #               output_dir_main + '\\' + 'KK', output_dir_main + '\\' + 'PP']

    output_dirs = [output_dir_main + '\\' + 'PP']

    #num_annots = [185, 901, 1362, 1749, 71]
    num_annots = [71]

    #neg_names = ['CB_negatives.xlsx', 'ULU_negatives.xlsx', 'ULU2022_negatives.xlsx',
    #             'KK_negatives.xlsx', 'PP_negatives.xlsx']

    neg_names = ['PP_negatives.xlsx']

    spec_file = r'C:\Users\kzammit\Documents\Detector\manual_dataset\spec_config_2sec.json'

    data_dir = r'D:\ringed-seal-data'

    file_durations = pd.read_excel(r'C:\Users\kzammit\Documents\Detector\manual_dataset\formatted_annots'
                                   r'\all_file_durations_complete.xlsx')

    #create_tables(annot_tables, main_folder, neg_names, file_durations, output_dirs, spec_file, data_dir)
    create_tables(annot_tables, main_folder, neg_names, file_durations, output_dirs, spec_file, data_dir, num_annots)








