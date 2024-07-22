import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.pyplot import figure

fig = figure(figsize=(10, 6), dpi=80)

df = pd.read_csv(r'E:\baseline-with-normalization-reduce-tonal\annots\pos\ULU2022_all_formatted_1sec.csv')

df['date'] = [x.split(".")[1] for x in df['filename']]

df['date2'] = [x[:6] for x in df['date']]

df = df.sort_values('date')

df_tr = df[0:1038]
df_te = df[1038:1334]
df_va = df[1334:]

#te_line = df.loc[1038]['date']
#va_line = df.loc[1334]['date']
#448
ax = sns.histplot(df_tr, x='date', bins=range(0,1000), color='#EFB118', edgecolor=None)
sns.histplot(df_te, x='date', bins=range(0,1000), color='#4269D0', edgecolor=None)
sns.histplot(df_va, x='date', bins=range(0,1000), color='#3CA951', edgecolor=None)
#ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
#ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

ax.set_xticks(ax.get_xticks()[::75])
plt.xticks(rotation=25)
plt.ylabel('# of Segments')
plt.xlabel('Date (yymmdd)')
fig.legend(labels=['Training','Validation','Testing'], bbox_to_anchor=(0.4, 0., 0.5, 0.88))
#plt.xticks(np.arange(min(df_tr['date']), max(df_va['date'])+1, 100.0))
#plt.axvline(x = te_line, color = 'b', label = 'axvline - full height')
#plt.axvline(x = va_line, color = 'b', label = 'axvline - full height')

plt.savefig(r'E:\annot_stats\ulu22-split-by-time.png', bbox_inches='tight', pad_inches=0.5)
plt.show()

print('test')