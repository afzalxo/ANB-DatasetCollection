import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

df = pd.read_csv('./Book1.csv')
df.columns = ['JID', 'Model Num','','', '', 'Latency (ms)', 'Throughput (fps)', 'MACs', 'MParams']
df.insert(0, 'Model Index', range(len(df)))
print(df)
#df = df[df['MParams'] > 8]

scatter = sns.scatterplot(data=df, x='MParams', y='Throughput (fps)', size='MACs', color=['k'])

#scatter = sns.scatterplot(data=df2, x='MParams', y='ImageNet Top-1 Acc', hue='Model Index', palette='Spectral', size='MACs', marker='o')
#scatter.legend_.remove()
plt.savefig('./thr.pdf')
#print(df)
