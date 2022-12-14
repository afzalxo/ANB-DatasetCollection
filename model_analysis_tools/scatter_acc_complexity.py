import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

df = pd.read_csv('./proxified_models_37.csv')
df.columns = ['JID', 'Model Num', 'ImageNet Top-1 Acc', 'Acc Top-5', 'MACs', 'MParams', 'Train Time']
df['Train Time'] = -1*df['Train Time']
df['Training Regime'] = 'Proxified'
df.insert(0, 'Model Index', range(len(df)))

df2 = pd.read_csv('./exact_models_37.csv')
df2.columns = ['JID', 'Model Num', 'ImageNet Top-1 Acc', 'Acc Top-5', 'MACs', 'MParams', 'Train Time']
df2['Train Time'] = -1*df2['Train Time']
df2['Training Regime'] = 'Exact'
df2.insert(0, 'Model Index', range(len(df2)))

df['Ranks'] = df['ImageNet Top-1 Acc'].rank(ascending=False)
df2['Ranks'] = df2['ImageNet Top-1 Acc'].rank(ascending=False)

scatter = sns.scatterplot(data=df, x='MACs', y='ImageNet Top-1 Acc', hue='Model Index', palette='Spectral', size='MParams', marker='D')
scatter = sns.scatterplot(data=df2, x='MACs', y='ImageNet Top-1 Acc', hue='Model Index', palette='Spectral', size='MParams', marker='o')
scatter.legend_.remove()
plt.savefig('./exact.pdf')
print(df)

kt = kendalltau(df['Ranks'], df2['Ranks'])
print(kt)
