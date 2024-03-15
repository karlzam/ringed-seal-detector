import pandas as pd
import numpy as np
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

folder = r'E:\baseline-with-normalization-reduce-tonal\logs'

files = glob.glob(folder + '\*.csv')

accuracy_tr = pd.DataFrame(columns=['epoch'])
precision_tr = pd.DataFrame(columns=['epoch'])
recall_tr = pd.DataFrame(columns=['epoch'])
loss_tr = pd.DataFrame(columns=['epoch'])

accuracy_val = pd.DataFrame(columns=['epoch'])
precision_val = pd.DataFrame(columns=['epoch'])
recall_val = pd.DataFrame(columns=['epoch'])
loss_val = pd.DataFrame(columns=['epoch'])

for idx, file in enumerate(files):

    csv_file = pd.read_csv(file)
        
    train_df = csv_file[csv_file['dataset'] == 'train']
    val_df = csv_file[csv_file['dataset'] == 'val']
    
    if idx == 0:
        accuracy_tr['epoch'] = train_df['epoch']
        precision_tr['epoch'] = train_df['epoch']
        recall_tr['epoch'] = train_df['epoch']
        loss_tr['epoch'] = train_df['epoch']
        
        accuracy_val['epoch'] = val_df['epoch']
        precision_val['epoch'] = val_df['epoch']
        recall_val['epoch'] = val_df['epoch']
        loss_val['epoch'] = val_df['epoch']
    
    model_name = file.split("\\")[-1].split('.')[0]
    
    accuracy_tr[model_name] = train_df['CategoricalAccuracy'].tolist()
    precision_tr[model_name] = train_df['Precision'].tolist()
    recall_tr[model_name] = train_df['Recall'].tolist()
    loss_tr[model_name] = train_df['loss'].tolist()
    
    accuracy_val[model_name] = val_df['CategoricalAccuracy'].tolist()
    precision_val[model_name] = val_df['Precision'].tolist()
    recall_val[model_name] = val_df['Recall'].tolist()
    loss_val[model_name] = val_df['loss'].tolist()

accuracy_tr['mean'] = accuracy_tr.loc[:, accuracy_tr.columns != 'epoch'].mean(axis=1)
accuracy_tr['std'] = accuracy_tr.loc[:, accuracy_tr.columns != 'epoch'].std(axis=1)

precision_tr['mean'] = precision_tr.loc[:, precision_tr.columns != 'epoch'].mean(axis=1)
precision_tr['std'] = precision_tr.loc[:, precision_tr.columns != 'epoch'].std(axis=1)

recall_tr['mean'] = recall_tr.loc[:, recall_tr.columns != 'epoch'].mean(axis=1)
recall_tr['std'] = recall_tr.loc[:, recall_tr.columns != 'epoch'].std(axis=1)

loss_tr['mean'] = loss_tr.loc[:, loss_tr.columns != 'epoch'].mean(axis=1)
loss_tr['std'] = loss_tr.loc[:, loss_tr.columns != 'epoch'].std(axis=1)

accuracy_val['mean'] = accuracy_val.loc[:, accuracy_val.columns != 'epoch'].mean(axis=1)
accuracy_val['std'] = accuracy_val.loc[:, accuracy_val.columns != 'epoch'].std(axis=1)

precision_val['mean'] = precision_val.loc[:, precision_val.columns != 'epoch'].mean(axis=1)
precision_val['std'] = precision_val.loc[:, precision_val.columns != 'epoch'].std(axis=1)

recall_val['mean'] = recall_val.loc[:, recall_val.columns != 'epoch'].mean(axis=1)
recall_val['std'] = recall_val.loc[:, recall_val.columns != 'epoch'].std(axis=1)

loss_val['mean'] = loss_val.loc[:, loss_val.columns != 'epoch'].mean(axis=1)
loss_val['std'] = loss_val.loc[:, loss_val.columns != 'epoch'].std(axis=1)

plt.plot(accuracy_tr['epoch'], accuracy_tr['mean'], '#377eb8', label='Train')
plt.fill_between(accuracy_tr['epoch'], accuracy_tr['mean'] - accuracy_tr['std'],
                 accuracy_tr['mean'] + accuracy_tr['std'], color='#377eb8', alpha=0.2)


plt.plot(accuracy_val['epoch'], accuracy_val['mean'], '#ff7f00', label='Val')
plt.fill_between(accuracy_val['epoch'], accuracy_val['mean'] - accuracy_val['std'],
                 accuracy_val['mean'] + accuracy_val['std'], color='#ff7f00', alpha=0.2)

plt.title("Average Accuracy Curve with Shaded Standard Deviation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(folder + "\\" + 'average-accuracy.png')
plt.close()

plt.plot(loss_tr['epoch'], loss_tr['mean'], '#377eb8', label='Train')
plt.fill_between(loss_tr['epoch'], loss_tr['mean'] - loss_tr['std'],
                 loss_tr['mean'] + loss_tr['std'], color='#377eb8', alpha=0.2)


plt.plot(loss_val['epoch'], loss_val['mean'], '#ff7f00', label='Val')
plt.fill_between(loss_val['epoch'], loss_val['mean'] - loss_val['std'],
                 loss_val['mean'] + loss_val['std'], color='#ff7f00', alpha=0.2)

plt.title("Average Loss Curve with Shaded Standard Deviation")
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig(folder + "\\" + 'average-loss.png')
plt.close()

plt.plot(recall_tr['epoch'], recall_tr['mean'], '#377eb8', label='Train')
plt.fill_between(recall_tr['epoch'], recall_tr['mean'] - recall_tr['std'],
                 recall_tr['mean'] + recall_tr['std'], color='#377eb8', alpha=0.2)


plt.plot(recall_val['epoch'], recall_val['mean'], '#ff7f00', label='Val')
plt.fill_between(recall_val['epoch'], recall_val['mean'] - recall_val['std'],
                 recall_val['mean'] + recall_val['std'], color='#ff7f00', alpha=0.2)

plt.title("Average Recall Curve with Shaded Standard Deviation")
plt.xlabel("Epoch")
plt.ylabel("recall")
plt.legend()
plt.savefig(folder + "\\" + 'average-recall.png')
plt.close()

plt.plot(precision_tr['epoch'], precision_tr['mean'], '#377eb8', label='Train')
plt.fill_between(precision_tr['epoch'], precision_tr['mean'] - precision_tr['std'],
                 precision_tr['mean'] + precision_tr['std'], color='#377eb8', alpha=0.2)


plt.plot(precision_val['epoch'], precision_val['mean'], '#ff7f00', label='Val')
plt.fill_between(precision_val['epoch'], precision_val['mean'] - precision_val['std'],
                 precision_val['mean'] + precision_val['std'], color='#ff7f00', alpha=0.2)

plt.title("Average Precision Curve with Shaded Standard Deviation")
plt.xlabel("Epoch")
plt.ylabel("precision")
plt.legend()
plt.savefig(folder + "\\" + 'average-precision.png')
plt.close()