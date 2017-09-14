import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Open','High','Low','Close','Volume']]
df['HL_PCT'] = (df['High'] - df['Close'])/df['Close'] * 100.0   #Percentage change on high and close values
df['PCT_change'] = (df['Close'] - df['Open'])/df['Close'] * 100.0  #Percentage change on close and open values

df = df[['Close','HL_PCT','PCT_change','Volume']]

forecast_col = 'Close' #Attribute you want to predict(which acts as a label)

df.fillna(-99999,inplace=True) #This fills up the null values with -99999 value

"""
There are 3226 rows, by doing the following
we are trying to predict next day closing price
"""
forecast_out = int(math.ceil(0.0001 * len(df)))
"""
This makes sure that previous day's prediction is equal to the next day's closing price
"""
df['label'] = df[forecast_col].shift(-forecast_out) #This shifts up
df.dropna(inplace=True)

#No.of days in future you want to predict
no_of_days = 10

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])
X = preprocessing.scale(X);
X_current = X
X = X_current[:-no_of_days]
y = y[:-no_of_days]
X_future = X_current[-no_of_days:] #features for prediction

#Spliting data to train and test
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2);

#initiating a classifier
clf = LinearRegression()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
forecast_set = clf.predict(X_future)
print(forecast_set,accuracy,no_of_days,'\n')
df = df[:-no_of_days]
df['forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_unix] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Close'].plot()
df['forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Dates')
plt.ylabel('Price')
plt.show()
