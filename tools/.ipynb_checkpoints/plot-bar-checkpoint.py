import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##
data = pd.read_excel(r'E:\baseline-with-normalization-reduce-tonal\detections-for-pp-ulu.xlsx')
output_dir = r'E:\baseline-with-normalization-reduce-tonal'
f, axarr = plt.subplots(2, sharex=True)
colors = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677"]
# Set your custom color palette
sns.set_palette(sns.color_palette(colors))
sns.barplot(data, x="Threshold", y="Ulu2023", hue="Detection Type", ax=axarr[0])
sns.barplot(data, x="Threshold", y="Pearce Point", hue="Detection Type", ax=axarr[1])
axarr[1].get_legend().remove()
sns.move_legend(axarr[0], "upper left", bbox_to_anchor=(1, 1))
plt.savefig(output_dir + '\\' + 'barplot-split-fp-types.png', bbox_inches='tight')
#plt.show()

##
f, axarr = plt.subplots(2, sharex=True)
data = pd.read_excel(r'E:\baseline-with-normalization-reduce-tonal\detections-for-pp-ulu-tp-tn.xlsx')
output_dir = r'E:\baseline-with-normalization-reduce-tonal'
colors = ["#332288", "#44AA99"]
# Set your custom color palette
sns.set_palette(sns.color_palette(colors))
sns.barplot(data, x="Threshold", y="Ulu2023", hue="Detection Type", ax=axarr[0])
sns.barplot(data, x="Threshold", y="Pearce Point", hue="Detection Type", ax=axarr[1])
axarr[1].get_legend().remove()
sns.move_legend(axarr[0], "upper left", bbox_to_anchor=(1, 1))
plt.savefig(output_dir + '\\' + 'barplot-tp-fp.png', bbox_inches='tight')
plt.show()