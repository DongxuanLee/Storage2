from pandas import DataFrame
from math import sqrt
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from datetime import datetime as dt
import sys
import csv
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot

def parser(x):
	return datetime.strptime(x, '%Y-%m-%d')
	#return datetime.strptime('190'+x, '%Y-%m')
filename = '2013.csv'

series = read_csv(filename, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

order=(3,0,3)


X = series.values
size = int(len(X) * 0.4)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	#print(history)
	model = ARIMA(history, order=order)
	#print(test[t])
	model_fit = model.fit(disp=0,trend='c')
	output = model_fit.forecast()
	yhat = output[0]
	print(output)
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))

error = sqrt(mean_squared_error(test, predictions))
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()