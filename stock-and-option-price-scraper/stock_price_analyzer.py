import pandas as pd
import json

from stockmodule import Stock
from stockmodule import Volatility

import seaborn as sns

from scipy.stats import shapiro
from scipy.stats import norm

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from matplotlib import pyplot
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot

import numpy as np

path = "C:\\Users\\orent\\Documents\\StockDataDownloader\\2020-08-12_23_07_one_time_run\\data.json"

with open(path, "r") as f:
    data = json.load(f)
s = Stock.from_dict(data['MSFT'])

stock_price_data = s.stock_price_history
stock_price_data = stock_price_data[ ["Close"] ]
#stock_price_data = stock_price_data.tail(180)

v = Volatility(stock_price_data, "Close")
#q, p = shapiro(v.time_series_daily['change'])
#qqplot(v.time_series_daily['change'], line='s')
#pyplot.show()

plt.hist(v.time_series_daily['change_log'], bins=250, density=True)
#plt.show()

mu, std = norm.fit(v.time_series_daily['change_log'])
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)

plt.show()

params = {'bandwidth': np.linspace(-0.1, 0.1, 20)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(v.time_series_daily['change'].to_numpy().reshape(-1,1))
grid.best_estimator_.bandwidth
kde = grid.best_estimator_

kde = KernelDensity(bandwidth=1.0)
kde.fit(v.time_series_daily['change'].to_numpy().reshape(-1,1))

test = kde.sample(100000)
sns.distplot(test, kde=False, norm_hist=True)
plt.show()

#sns.kdeplot(v.time_series_daily['change_log'], shade=True);
sns.distplot(v.time_series_daily['change'], kde=False, norm_hist=True)
plt.show()

v.get_return_descriptive_statistics()

v.plot_realized_volatility('rolling', [5, 20, 60])


#stock_price_data.sort_index(ascending=True, inplace=True)

#stock_price_data["tomorrows_close"] = stock_price_data["Close"].shift(-1)
#stock_price_data["daily_change"] = stock_price_data.apply(lambda row: (row["tomorrows_close"] - row["Close"]) / row["Close"], axis=1 )

#stock_price_data.std()
#stock_price_data.mean()