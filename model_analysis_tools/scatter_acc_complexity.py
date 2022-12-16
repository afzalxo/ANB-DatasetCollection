import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

df = pd.read_csv('./proxified_models_37.csv')
df.columns = ['JID', 'Model Num', 'ImageNet Top-1 Acc', 'Acc Top-5', 'MACs', 'MParams', 'Train Time']
df['Train Time'] = -1*df['Train Time']
df['Training Regime'] = 'Proxified'
df.insert(0, 'Model Index', range(len(df)))

#df = df[df['MParams'] > 8]

df2 = pd.read_csv('./exact_models_37.csv')
df2.columns = ['JID', 'Model Num', 'ImageNet Top-1 Acc', 'Acc Top-5', 'MACs', 'MParams', 'Train Time']
df2['Train Time'] = -1*df2['Train Time']
df2['Training Regime'] = 'Exact'
df2.insert(0, 'Model Index', range(len(df2)))

#df2 = df2[df2['MParams'] > 8]

df['Ranks'] = df['ImageNet Top-1 Acc'].rank(ascending=False)
df2['Ranks'] = df2['ImageNet Top-1 Acc'].rank(ascending=False)

scatter = sns.scatterplot(data=df, x='MParams', y='MACs', hue='Model Index', palette='Spectral', size='MACs', marker='D')
#scatter = sns.scatterplot(data=df2, x='MParams', y='ImageNet Top-1 Acc', hue='Model Index', palette='Spectral', size='MACs', marker='o')
scatter.legend_.remove()
plt.savefig('./exact.pdf')
#print(df)
print(df2)

kt = kendalltau(df['Ranks'], df2['Ranks'])
print(kt)

df = df.sort_values(by=['ImageNet Top-1 Acc'], ascending=False)
df2 = df2.sort_values(by=['ImageNet Top-1 Acc'], ascending=False)

aa = df[['Model Index', 'ImageNet Top-1 Acc', 'Ranks']]

pruned = aa.head(10)
ss = df2.loc[df2['Model Index'].isin(pruned['Model Index'])]
ss = ss[['Model Index', 'ImageNet Top-1 Acc', 'Ranks']]
print(pruned)
print(ss)

