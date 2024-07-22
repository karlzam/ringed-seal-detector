import pandas as pd
import matplotlib.pyplot as plt

output_dir = r'E:\baseline-with-normalization-reduce-tonal\pearce-point\fine-tune'

thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

#all_counts = pd.DataFrame(columns=["thresh", "BS", "RS", "ICE", "SC", "KN", "WA", "BO"])
all_counts = pd.DataFrame(columns=["thresh", "RS", "ICE", "KN", "SC", "WA", "BS"])

for thresh in thresholds:

    class_file = pd.read_csv(r'E:\baseline-with-normalization-reduce-tonal\pearce-point\fine-tune\raven-formatted'
                             r'-detections-fine-tune.txt', sep='\t', encoding='latin1')

    cats = class_file['Class'].unique()

    counts = []
    class_file_edited = class_file[class_file['score']>=thresh]
    for cat in cats:

        cat1 = class_file_edited[class_file_edited['Class']==cat]
        counts.append(len(cat1))

    append_list = [thresh, counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]]
    all_counts.loc[len(all_counts)] = append_list

    myexplode = [0.1, 0, 0, 0, 0, 0]

    wedges, texts, autotexts = plt.pie(counts, autopct='%1.1f%%', textprops=dict(color="w"), explode=myexplode)
    #wedges, texts, autotexts = plt.pie(counts, autopct='%1.1f%%', textprops=dict(color="w"))

    plt.legend(wedges, cats,
              title="Categories",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.title('Detections at Threshold ' + str(thresh))
    plt.tight_layout()
    plt.savefig(output_dir + '\\' + 'threshold' + str(thresh) + '.png')
    plt.close()


all_counts.to_excel(output_dir + '\\' + 'DETECTIONS-CLASSES.xlsx')

print('test')
