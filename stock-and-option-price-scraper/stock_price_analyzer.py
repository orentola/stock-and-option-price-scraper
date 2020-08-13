import pandas as pd

from stockmodule import Stock

path = "C:\\Users\\orent\\Documents\\StockDataDownloader\\2020-08-12_23_07_one_time_run\\data.json"

class Volatility(self):


with open(path, "r") as f:
    data = json.load(f)
s = Stock.from_dict(data['MSFT'])

stock_price_data = s.stock_price_history
stock_price_data = stock_price_data[ ["Close"] ]

stock_price_data.sort_index(ascending=True, inplace=True)

stock_price_data["tomorrows_close"] = stock_price_data["Close"].shift(-1)
stock_price_data["daily_change"] = stock_price_data.apply(lambda row: (row["tomorrows_close"] - row["Close"]) / row["Close"], axis=1 )

stock_price_data.std()
stock_price_data.mean()