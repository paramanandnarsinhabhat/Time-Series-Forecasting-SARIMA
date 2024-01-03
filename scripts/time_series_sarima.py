import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 


from sklearn.metrics import mean_squared_error
from math import sqrt
from statistics import mean 

import warnings

train_data = pd.read_csv("/Users/paramanandbhat/Downloads/7.3_ARIMA_and_SARIMA_models/data/train_data.csv")
valid_data = pd.read_csv("/Users/paramanandbhat/Downloads/7.3_ARIMA_and_SARIMA_models/data/valid_data.csv")


print(train_data.shape)
print(train_data.head())

print(valid_data.shape)
print(valid_data.head())

# Required Preprocessing 
train_data.timestamp = pd.to_datetime(train_data['Date'],format='%Y-%m-%d')
train_data.index = train_data.timestamp

valid_data.timestamp = pd.to_datetime(valid_data['Date'],format='%Y-%m-%d')
valid_data.index = valid_data.timestamp

plt.figure(figsize=(12,8))

plt.plot(train_data.index, train_data['count'], label='train_data')
plt.plot(valid_data.index,valid_data['count'], label='valid')
plt.legend(loc='best')
plt.title("Train and Validation Data")
plt.show()

# Stationarity Test
# dickey fuller, KPSS
from statsmodels.tsa.stattools import adfuller, kpss

def adf_test(timeseries):
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput=pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

adf_test(train_data['count'])
   
'''
If the test statistic is less than the 
critical value, we can reject the null 
hypothesis (aka the series is stationary). 
When the test statistic is greater than the
 critical value, we fail to reject the null 
 hypothesis (which means the series is not 
 stationary). **
Here test statistic is > than critical. 
Hence series is not stationary**

'''
def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)


kpss_test(train_data['count'])

'''
If the test statistic is greater than the critical value, we reject the null hypothesis (series is not stationary). If the test statistic is less than the critical value, if fail to reject the null hypothesis (series is stationary).  **Here test statistic is > than critical. Hence series is not stationary**

Alternatively, we can use the p-value to make the inference. If p-value is less than 0.05, we can reject the null hypothesis. And say that the series is not stationary.

'''

# Making Series Stationary

train_data['count_diff'] = train_data['count'] - train_data['count'].shift(1)

plt.figure(figsize=(12,8))

plt.plot(train_data.index, train_data['count'], label='train_data')
plt.plot(train_data.index,train_data['count_diff'], label='stationary series')
plt.legend(loc='best')
plt.title("Stationary Series")
plt.show()

train_data['count_log'] = np.log(train_data['count'])
train_data['count_log_diff'] = train_data['count_log'] - train_data['count_log'].shift(1)

plt.figure(figsize=(12,8))

plt.plot(train_data.index,train_data['count_log_diff'], label='stationary series')
plt.legend(loc='best')
plt.title("Stationary Series")
plt.show()

adf_test(train_data['count_log_diff'].dropna())

kpss_test(train_data['count_log_diff'].dropna())

# ACF and PACF plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(train_data['count_log_diff'].dropna(), lags=15)
plot_pacf(train_data['count_log_diff'].dropna(), lags=15)
plt.show()

'''
   - p value is the lag value where the PACF chart crosses the confidence interval for the first time. It can be noticed that in this case p=2.

   - q value is the lag value where the ACF chart crosses the confidence interval for the first time. It can be noticed that in this case q=2.

   - Now we will make the ARIMA model as we have the p,q values.

'''

# SARIMA
from statsmodels.tsa.statespace import sarimax

plot_acf(train_data['count_log_diff'].dropna(), lags=25)
plot_pacf(train_data['count_log_diff'].dropna(), lags=25)
plt.show()

train_data['count_log'] = np.log(train_data['count'])
train_data['count_log_diff'] = train_data['count_log'] - train_data['count_log'].shift(7)

train_data['count_log_diff'].head(10)

plot_acf(train_data['count_log_diff'].dropna(), lags=25)
plot_pacf(train_data['count_log_diff'].dropna(), lags=25)
plt.show()

# fit model
model = sarimax.SARIMAX(train_data['count_log'], seasonal_order=(1,1,1,7), order=(2,1,2))
fit1 = model.fit()

# make predictions
valid_data['SARIMA'] = fit1.predict(start="2014-02-09", end="2014-09-25", dynamic=True)


valid_data['SARIMA'] = np.exp(valid_data['SARIMA'])

plt.figure(figsize=(12,8))

plt.plot(train_data['count'],  label='train') 
plt.plot(valid_data['count'],  label='valid') 
plt.plot(valid_data['SARIMA'],  label='predicted') 
plt.legend(loc='best') 
plt.show()

