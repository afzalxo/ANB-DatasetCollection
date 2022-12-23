import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

df = pd.read_csv('./junk.csv')
df.columns = ['JID', 'Model Num', 'ImageNet Top-1 Acc', 'Acc Top-5', 'MACs', 'MParams', 'Train Time']
df['Train Time'] = -1*df['Train Time']
df['Training Regime'] = 'Proxified'
df.insert(0, 'Model Index', range(len(df)))

#df = df[df['MParams'] > 8]

df['Ranks'] = df['ImageNet Top-1 Acc'].rank(ascending=False)

scatter = sns.scatterplot(data=df, x='MParams', y='ImageNet Top-1 Acc', hue='Train Time', size='MACs', marker='o')
#scatter.legend_.remove()
plt.savefig('./exact.pdf')
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
