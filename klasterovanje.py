from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler


df = pd.read_csv('bank.csv')
print(df)

lenc = LabelEncoder()
df['id'] = lenc.fit_transform(df['id'])
df['sex'] = lenc.fit_transform(df['sex'])
df['region'] = lenc.fit_transform(df['region'])
df['married'] = lenc.fit_transform(df['married'])
df['car'] = lenc.fit_transform(df['car'])
df['save_act'] = lenc.fit_transform(df['save_act'])
df['current_act'] = lenc.fit_transform(df['current_act'])
df['mortgage'] = lenc.fit_transform(df['mortgage'])
df['pep'] = lenc.fit_transform(df['pep'])


print(df)
X = df[['id', 'age', 'sex', 'region', 'income', 'married', 'children', 'car', 'save_act', 'current_act', 'mortgage', 'pep']]

#X = df[['age', 'income']]

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
#dbscan = DBSCAN(eps=0.02, min_samples=5)
#dbscan.fit(X)
#labels = dbscan.labels_

plt.scatter(X['age'], X['income'], c=labels)
plt.show()
