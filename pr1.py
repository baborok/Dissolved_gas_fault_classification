import matplotlib.pyplot as plt
import pandas as pd
import datetime 
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

data = pd.read_csv(r"C:\Users\petuu\Desktop\Transformer-Health-Index-Indication-Classifier-main/sub.csv")

data = data.set_index('date')

#data = data.rename(columns={'index': 'y'})

#data = data.sort_index()
#data.head()

#print(data)

data.index = pd.to_datetime(data.index)

data.plot(style='.',
        figsize=(15, 5),
        color=color_pal[0],
        title='Индексы трансформатора на протяжении отрезка')
plt.show()

train = data.loc[data.index < '2023-04-13 15:16:07.275160']
test = data.loc[data.index >= '2023-04-13 15:16:07.275160']

#plt.plot(data, 'green', label='Текущее состояние')
#plt.plot(data_test, 'blue', label='Предсказание состояния')
#plt.show()

#plt.plot(data['index'])
#plt.show()

#model = ARIMA(data_train, order=(2, 1, 0)) 
#results = model.fit()
#forecast = results.forecast(steps=200)
#print(forecast)

#from sklearn.metrics import mean_squared_error, mean_absolute_error
#import numpy as np

#mean absolute error
#mae = mean_absolute_error(data_test, forecast)

#root mean square error
#mse = mean_squared_error(data_test, forecast)
#rmse = np.sqrt(mse)

#mean absolute percentage error


#print(f"MAE: {mae:.2f}")
#print(f"RMSE: {rmse:.2f}")


#plt.plot(data_train, label='Train')
#plt.plot(data_test, label='Test')
#plt.plot(forecast, label='Forecast')
#plt.legend()
#plt.show()

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Ретроспективные данные', title='Предсказание')
test.plot(ax=ax, label='Предсказание')
ax.axvline('2023-04-13 15:16:07.275160', color='black', ls='--')
ax.legend(['Ретроспективные данные', 'Предсказание'])
plt.show()


def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['second'] = df.index.second
    df['minute'] = df.index.minute
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(data)

print(df)


fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='second', y='index')
ax.set_title('Изменения в секундах')
plt.show()

train = create_features(train)
test = create_features(test)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year','minute','second']
TARGET = 'index'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)

test['prediction'] = reg.predict(X_test)
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
ax = df[['index']].plot(figsize=(15, 5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Реальное состояние', 'Предсказания'])
ax.set_title('Предсказание и тренд')
plt.show()