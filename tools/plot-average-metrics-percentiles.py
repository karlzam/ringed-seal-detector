import pandas as pd
import glob
import matplotlib.pyplot as plt

folder = r'E:\baseline-with-normalization-reduce-tonal\metrics\ensemble\percentiles\stats'

files = glob.glob(folder + '\*.csv')

precision = pd.DataFrame(columns=['threshold'])
recall = pd.DataFrame(columns=['threshold'])
f1 = pd.DataFrame(columns=['threshold'])

for idx, file in enumerate(files):

    csv_f = pd.read_csv(file)

    if idx == 0:
        precision['threshold'] = csv_f['threshold']
        recall['threshold'] = csv_f['threshold']
        f1['threshold'] = csv_f['threshold']

    model_name = file.split("\\")[-1].split('.')[0]
    precision[model_name] = csv_f['precision'].tolist()
    recall[model_name] = csv_f['recall'].tolist()
    f1[model_name] = csv_f['f1'].tolist()

plt.plot(precision['threshold'], precision['stats-mean'], '#377eb8', label='Precision')
plt.fill_between(precision['threshold'], precision['stats-10'], precision['stats-90'], color='#377eb8', alpha=0.2)

plt.plot(recall['threshold'], recall['stats-mean'], '#ff7f00', label='Recall')
plt.fill_between(recall['threshold'], recall['stats-10'], recall['stats-90'], color='#ff7f00', alpha=0.2)

#plt.plot(f1['threshold'], f1['stats-mean'], '#999999', label='F1')
#plt.fill_between(f1['threshold'], f1['stats-10'], f1['stats-90'], color='#999999', alpha=0.2)

plt.legend()
#plt.title('Average Metrics Across 10 Model Runs with Shaded Standard Deviation')
plt.xlabel('Threshold')
plt.ylabel('Value')
plt.savefig(folder + '\\' + 'percentile_metrics-mean.png')
plt.close()

plt.plot(precision['threshold'], precision['stats-50'], '#377eb8', label='Precision')
plt.fill_between(precision['threshold'], precision['stats-10'], precision['stats-90'], color='#377eb8', alpha=0.2)

plt.plot(recall['threshold'], recall['stats-50'], '#ff7f00', label='Recall')
plt.fill_between(recall['threshold'], recall['stats-10'], recall['stats-90'], color='#ff7f00', alpha=0.2)

#plt.plot(f1['threshold'], f1['stats-mean'], '#999999', label='F1')
#plt.fill_between(f1['threshold'], f1['stats-10'], f1['stats-90'], color='#999999', alpha=0.2)

plt.legend()
#plt.title('Average Metrics Across 10 Model Runs with Shaded Standard Deviation')
plt.xlabel('Threshold')
plt.ylabel('Value')
plt.savefig(folder + '\\' + 'percentile_metrics-median.png')

#print('precision is ' + str(float(precision[precision['threshold']==0.55]['mean'])) + 'std of ' + str(float(precision[precision['threshold']==0.55]['std'])))
#print('recall is ' + str(float(recall[recall['threshold']==0.55]['mean'])) + 'std of ' + str(float(recall[recall['threshold']==0.55]['std'])))
