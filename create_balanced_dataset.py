import pandas as pd
from sklearn.utils import shuffle

df = pd.read_csv("ddos_dataset.csv")
print("load data ok")
cnts = df['Label'].value_counts()
print(cnts)
benign = cnts[0]
print(benign)
by_class = df.groupby('Label')

datasets = {}
for groups, data in by_class:
    datasets[groups] = data
a = datasets[1]
b = datasets[0]

smple = a.sample(n=benign)
balance_data = b.append([smple])
balance_data = shuffle(balance_data)
print("balance_data ok")

balance_data.to_csv("balanced_dataset.csv", index=False)
print("finish")
