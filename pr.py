import matplotlib.pyplot as plt
import pandas as pd
import datetime 
import sklearn as sk


data = pd.read_csv(r"C:\Users\petuu\Desktop\Transformer-Health-Index-Indication-Classifier-main/sub.csv")

data = data.set_index('date')

data = data.rename(columns={'index': 'y'})

data = data.sort_index()
data.head()

print(data)

steps = 1000
data_train = data[:-steps]
data_test  = data[-steps:]

print(f"Train dates : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
print(f"Test dates  : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

#fig, ax = plt.subplots(figsize=(100, 30))
#data_train['y'].plot(ax=ax, label='train')
#data_test['y'].plot(ax=ax, label='test')
#ax.legend()
#plt.show()

plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Даты')
plt.ylabel('Индекс трансформатора')
plt.plot(data, 'green', label='Текущее состояние')
plt.plot(data_test, 'blue', label='Предсказание состояния')
plt.legend()
plt.show()


