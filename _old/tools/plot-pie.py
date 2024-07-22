import pandas as pd
import matplotlib.pyplot as plt

output_dir = r'E:\baseline-with-normalization-reduce-tonal\deploy\ulu2023\fine-tuning\deploy_on_audio'

thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

for thresh in thresholds:

    class_file = pd.read_excel(r'E:\baseline-with-normalization-reduce-tonal\deploy\ulu2023\fine-tuning'
                               r'\deploy_on_audio\raven-formatted-detections.xlsx')

    cats = class_file['Class'].unique()

    counts = []
    class_file_edited = class_file[class_file['score']>=thresh]
    for cat in cats:

        cat1 = class_file_edited[class_file_edited['Class']==cat]
        counts.append(len(cat1))

    myexplode = [0, 0, 0.2, 0, 0, 0, 0, 0]

    wedges, texts, autotexts = plt.pie(counts, autopct='%1.1f%%', textprops=dict(color="w"), explode=myexplode)

    plt.legend(wedges, cats,
              title="Categories",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.title('Scores Greater Than or Equal to ' + str(thresh))
    plt.tight_layout()
    plt.savefig(output_dir + '\\' + 'threshold' + str(thresh) + '.png')
    plt.close()
