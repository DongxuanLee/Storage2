import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf ,pacf
from statsmodels.tsa.arima_model import ARIMA
def main():
    year = input('input the year you want to look:')
    filename = ('{}.csv'.format(year))

    dataframe = read_file(filename)
    sdate = dataframe[2]
    new_sdate = []
    flag = dataframe[9]
    flag = np.asarray(flag).astype(float).astype(int)
    goods = dataframe[14]
    goods = np.asarray(goods)
    num = dataframe[21]
    num = abs(np.asarray(num).astype(float))
    new_num = []
    for i in range(len(sdate)):
        if goods[i] == '2996' and flag[i] == 3:
            new_num.append(num[i])
            tmp = sdate[i].split(' ')
            if len(tmp) != 0:
                new_sdate.append(tmp[0])
            else:
                new_sdate.append(0)
                print('DATE SPLIT ERROR!')
    data = Date_Differ(new_sdate,new_num)

    f_data= DataFrame(data,columns=['date', 'num'])
    f_data['num'] = f_data['num'].convert_objects(convert_numeric=True)

    #plt.plot(f_data['num'][1:])
    #plt.show()
    # test_stationarity(f_data[1])
    #plt.show()
    # estimating(f_data[1])
    Diff(f_data['num'])
    # Decomposing(f_data[1])
    # print(type(f_data['num']))
    # ma = ARIMA_Model(f_data['num'])
    # print(ma)
def read_file(filename):
    dataframe = pd.read_csv(filename,usecols=[2,9,14,21],names=None,encoding='ISO-8859-1',header=None,low_memory=False)
    dataframe = dataframe[1:]
    return dataframe

def test_stationarity(timeseries):
    # print(timeseries)
    # print("------------")
    rolmean = timeseries.rolling(30).mean()
    # print(rolmean)
    rol_weighted_mean = timeseries.ewm(30).mean()
    rolstd = timeseries.rolling(30).std()
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    weighted_mean = plt.plot(rol_weighted_mean, color='green', label='weighted Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    # 进行df测试
    print('Result of Dickry-Fuller test')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value(%s)' % key] = value
    print(dfoutput)

def Diff(timeseries):
    ts_log_diff = np.log(timeseries).diff(1)
    # ts_log_diff2 = ts_log_diff.diff(1)
    # print(ts_log_diff2)
    # plt.plot(ts_log_diff1,label = 'diff 1')
    # plt.plot(ts_log_diff2, label='diff 2')
    # plt.legend(loc = 'best')
    ts_log_diff.dropna(inplace=True)
    # ts_log_diff2.dropna(inplace=True)
    test_stationarity(ts_log_diff)
    plt.show()
    # 确定参数
    lag_acf = acf(ts_log_diff, nlags=20)

    lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

    # q的获取:ACF图中曲线第一次穿过上置信区间
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')  # lowwer置信区间
    plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')  # upper置信区间
    plt.title('Autocorrelation Function')
    # p的获取:PACF图中曲线第一次穿过上置信区间
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()
# def Decomposing(timeseries):
#     ts_log = np.log(timeseries)
#     decomposition = seasonal_decompose(ts_log,freq=30)
#
#     trend = decomposition.trend  # 趋势
#     seasonal = decomposition.seasonal  # 季节性
#     residual = decomposition.resid  # 剩余的
#
#     plt.subplot(411)
#     plt.plot(ts_log, label='Original')
#     plt.legend(loc='best')
#     plt.subplot(412)
#     plt.plot(trend, label='Trend')
#     plt.legend(loc='best')
#     plt.subplot(413)
#     plt.plot(seasonal, label='Seasonarity')
#     plt.legend(loc='best')
#     plt.subplot(414)
#     plt.plot(residual, label='Residual')
#     plt.legend(loc='best')
#     plt.tight_layout()
#     plt.show()
def ARIMA_Model(timeseries):
    ts_log = np.log(timeseries)
    print(type(ts_log[0]))
    model = ARIMA(ts_log, order=(1, 1, 1))
    result_ARIMA = model.fit(disp=-1)
    ts_log_diff = Diff(timeseries)
    # plt.plot(ts_log_diff)
    # plt.plot(result_ARIMA.fittedvalues, color='red')
    # plt.title('MA model RSS:%.4f' % sum(result_ARIMA.fittedvalues - ts_log_diff) ** 2)
    # plt.show()

    predictions_ARIMA_diff = pd.Series(result_ARIMA.fittedvalues, copy=True)
    print(predictions_ARIMA_diff) #发现数据是没有第一行的,因为有1的延迟

    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    print(predictions_ARIMA_diff_cumsum)

    predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
    print(predictions_ARIMA_log)

    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    return predictions_ARIMA
    # plt.plot(ts)
    # plt.plot(predictions_ARIMA)
    # plt.title('predictions_ARIMA RMSE: %.4f' % np.sqrt(sum((predictions_ARIMA - ts) ** 2) / len(ts)))
    # plt.show()


def Date_Differ(date_list,num_list):
    sum = 0
    newdate = []
    newnum = []
    for i in range(len(date_list)):
        if i == 0:
            newdate.append(date_list[0])
            sum = num_list[0]
        else:
            if date_list[i] != date_list[i - 1]:
                newdate.append(date_list[i])
                newnum.append(sum)
                sum = 0
            sum = sum + num_list[i]
            if i == len(date_list) - 1:
                newnum.append(sum)
    date_num =np.asarray([])
    for i in range(len(newdate)):
        if i == 0:
            date_num=np.asarray([newdate[0],newnum[0]]).reshape((1,2))
        else:
            tmp_ = date_num
            date_num = np.vstack((tmp_,[newdate[i],newnum[i]]))
    #print(type(newnum[10]))
    return date_num

if __name__ == '__main__':
    main()

