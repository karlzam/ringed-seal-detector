import pandas as pd
import glob
import matplotlib.pyplot as plt

folder = r'E:\baseline-with-normalization-reduce-tonal\metrics\stats'

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

precision['mean'] = precision.loc[:, precision.columns != 'threshold'].mean(axis=1)
precision['std'] = precision.loc[:, precision.columns != 'threshold'].std(axis=1)
precision['mean+std'] = precision['mean'] + precision['std']
precision['mean-std'] = precision['mean'] - precision['std']

recall['mean'] = recall.loc[:, recall.columns != 'threshold'].mean(axis=1)
recall['std'] = recall.loc[:, recall.columns != 'threshold'].std(axis=1)
recall['mean+std'] = recall['mean'] + recall['std']
recall['mean-std'] = recall['mean'] - recall['std']

f1['mean'] = f1.loc[:, f1.columns != 'threshold'].mean(axis=1)
f1['std'] = f1.loc[:, f1.columns != 'threshold'].std(axis=1)
f1['mean+std'] = f1['mean'] + f1['std']
f1['mean-std'] = f1['mean'] - f1['std']

#dfm = precision.melt('threshold', var_name='cols', value_name='vals')
#g = sns.catplot(x="threshold", y="vals", hue='cols', data=dfm, kind='point')

plt.plot(precision['threshold'], precision['mean'], '#377eb8', label='Precision')
plt.fill_between(precision['threshold'], precision['mean-std'], precision['mean+std'], color='#377eb8', alpha=0.2)

plt.plot(recall['threshold'], recall['mean'], '#ff7f00', label='Recall')
plt.fill_between(recall['threshold'], recall['mean-std'], recall['mean+std'], color='#ff7f00', alpha=0.2)

plt.plot(f1['threshold'], f1['mean'], '#999999', label='F1')
plt.fill_between(f1['threshold'], f1['mean-std'], f1['mean+std'], color='#999999', alpha=0.2)

plt.legend()
plt.title('Average Metrics Across 10 Model Runs with Shaded Standard Deviation')
plt.xlabel('Threshold')
plt.ylabel('Value')
plt.savefig(folder + '\\' + 'average_metrics.png')

print('precision is ' + str(float(precision[precision['threshold']==0.55]['mean'])) + 'std of ' + str(float(precision[precision['threshold']==0.55]['std'])))
print('recall is ' + str(float(recall[recall['threshold']==0.55]['mean'])) + 'std of ' + str(float(recall[recall['threshold']==0.55]['std'])))

print('test')