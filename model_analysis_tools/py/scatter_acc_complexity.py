import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

parser = argparse.ArgumentParser('')
parser.add_argument('--csv_path', type=str, default=None, help='Path to dataset csv')
args = parser.parse_args()

df = pd.read_csv(args.csv_path)
df = df.drop(df.columns[list(range(8, 8+28))], axis=1)
df.columns = ['Junk', 'JID', 'Model Num', 'Acc Top-1', 'Acc Top-5', 'MACs', 'MParams', 'Train Time']
df['Training Regime'] = 'Proxified'
df.insert(0, 'Model Index', range(len(df)))

#df = df[df['MParams'] > 8]

df['Ranks'] = df['Acc Top-1'].rank(ascending=False)

scatter = sns.scatterplot(data=df, x='MParams', y='Acc Top-1', hue='Train Time', size='MACs', marker='o')
#scatter.legend_.remove()
plt.savefig('./scatter_acc_vs_complexity.pdf')
#print(df)

'''
df = df.sort_values(by=['ImageNet Top-1 Acc'], ascending=False)
df2 = df2.sort_values(by=['ImageNet Top-1 Acc'], ascending=False)

aa = df[['Model Index', 'ImageNet Top-1 Acc', 'Ranks']]

pruned = aa.head(10)
ss = df2.loc[df2['Model Index'].isin(pruned['Model Index'])]
ss = ss[['Model Index', 'ImageNet Top-1 Acc', 'Ranks']]
print(pruned)
print(ss)
'''
