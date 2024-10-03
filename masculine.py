import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

df = pd.read_csv('masculinity.csv')

#print(df.info())
#print(df.isnull().sum())

#print(df.head())

mapping_q1 = {
    'Very masculine':4,
    'Somewhat masculine':3,
    'Not very masculine':2,
    'Not at all masculine':1
}

mapping_q7 = {
'Often':5,
'Sometimes':4,
'Rarely':3,
'Never, but open to it':2,
'Never, and not open to it':1
}
df['q0001'] = df['q0001'].map(mapping_q1)
df['q0007_0006']=df['q0007_0006'].map(mapping_q7)
df['q0007_0007']=df['q0007_0007'].map(mapping_q7)
df['q0007_0009']=df['q0007_0009'].map(mapping_q7)

print(df['q0007_0006'])
columns_to_cluster = df[['q0001','q0007_0006','q0007_0007']]
columns_to_cluster = columns_to_cluster.dropna()
model = KMeans(n_clusters=2,random_state=42)
model.fit(columns_to_cluster)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with 3 features
ax.scatter(df['q0001'], df['q0007_0006'], df['q0007_0007'], c=model.labels_, cmap='viridis')


ax.set_xlabel('q0001')
ax.set_ylabel('q0007_0006')
ax.set_zlabel('q0007_0007')
plt.title('3D K-Means Clustering of Survey Responses')


plt.show()