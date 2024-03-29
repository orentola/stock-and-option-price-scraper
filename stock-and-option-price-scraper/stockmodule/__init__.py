import yfinance as yf
import pandas as pd
import os
import copy
import pytz
import json
import math

from statsmodels.graphics.tsaplots import plot_acf

import matplotlib.pyplot as plt

from json import JSONEncoder
from datetime import datetime
from time import sleep
from pathlib import Path

#############################
# Helper functions

class StockEncoder(JSONEncoder):
	def default(self, obj):
		if isinstance(obj, Stock):
			return stock_as_dict(obj)
		# If not handled, return default
		return JSONEncoder.default(self,obj)

class DownloaderEncoder(JSONEncoder):
	def default(self, obj):
		if isinstance(obj, Downloader):
			output_dict = {}
			for s in obj.stock_list:
				temp_json = stock_as_dict(s)
				output_dict[s.ticker] = temp_json
			return output_dict
		return JSONEncoder.default(self, obj)

def as_stock(dict):
	# No error handling at the moment
	stock_list = []
	for k1, v1 in dict.items():
		print("Creating new stock object for: " + v1["ticker"])
		stock_list.append(Stock.from_dict(v1))
	return stock_list

def stock_as_dict(obj):
	output_dict = {}
	output_dict["ticker"] = obj.ticker
	output_dict["data_endpoint"] = obj.data_endpoint
	output_dict["datetime_run"] = obj.datetime_run
	
	output_dict["stock_price_history"] = obj.stock_price_history.to_json()

	option_chain_list = []
	for it in obj.option_chains:
		option_chain_list.append(it.to_json())

	output_dict["option_chains"] = option_chain_list 
	return output_dict

###############################

class Downloader:
	def __init__(self, ticker_list_name):
		self.ticker_list_name = ticker_list_name
		self.ticker_list_path = os.path.join(os.getcwd(), 'tickers.txt')
		self.ticker_list = []
		self.stock_list = []

	def read_tickers(self):
		with open(self.ticker_list_path) as f:
			finished = False
			ticker_list_temp = []

			while (finished == False):
				line = f.readline()

				# EOF
				if len(line) == 0:
					finished = True
					break
				
				line = line.rstrip('\n')
				# Found a new set of tickers
				if line.find("#") > -1:
					# Check if we are at the list of tickers we want to be
					if line.replace("#", "") == self.ticker_list_name:
						# Read until end of section reached ('#')
						while (True):
							section_line = f.readline().rstrip('\n') 
							if (section_line == "#"):
								finished = True
								break
							ticker_list_temp.append(section_line)
			self.ticker_list = copy.deepcopy(ticker_list_temp)
	
	def download_stock_data(self, start_dt=None, end_dt=None):
		for s in self.stock_list:
			print("Downloading stock data for " + s.ticker + " with start_dt " + str(start_dt) + " and end_dt " + str(end_dt))
			s.download_stock_price_data(start_dt, end_dt)

	def download_options_data(self):
		for s in self.stock_list:
			print("Downloading options data for " + s.ticker + ".")
			s.download_options_data()

	def build_stock_objects(self):
		if (len(self.ticker_list) == 0):
			print('Ticker list is empty.')
		for t in self.ticker_list:
			self.stock_list.append(Stock(t))

	def initialize(self):
		self.read_tickers()
		self.build_stock_objects()


class OptionChain:
	def __init__(self, ticker, date, calls = None, puts = None, updated_time = None):
		self.ticker = ticker
		self.date = date
		self.calls = calls
		self.puts = puts
		self.updated_time = updated_time

	def to_json(self):
		output_dict = {}
		output_dict["ticker"] = self.ticker
		output_dict["date"] = self.date
		output_dict["calls"] = self.calls.to_json() if self.calls is not None else None
		output_dict["puts"] = self.puts.to_json() if self.puts is not None else None
		output_dict["updated_time"] = self.updated_time
		return output_dict
	
	@classmethod
	def from_dict(cls, dict):
		ticker = dict["ticker"]
		date = dict["date"] 
		#calls = pd.read_json(dict["calls"]) if dict["calls"] is not None else None
		#puts = pd.read_json(dict["puts"]) if dict["puts"] is not None else None
		calls = pd.read_json(dict["calls"]).set_index("contractSymbol", drop=True, inplace=False).rename_axis("contractSymbol") if dict["calls"] is not None else None
		puts = pd.read_json(dict["puts"]).set_index("contractSymbol", drop=True, inplace=False).rename_axis("contractSymbol") if dict["puts"] is not None else None
		updated_time = dict["updated_time"]
		return OptionChain(ticker, date, calls, puts, updated_time)

class Stock:
	def __init__(self, ticker, datetime_run = None, data_endpoint = "yfinance"):
		self.ticker = ticker
		self.stock_price_history = pd.DataFrame()
		self.option_chains = []
		self.yfinance_ticker_object = None
		self.data_endpoint = "yfinance"
		self.datetime_run = None

		self.update_object_based_on_data_endpoint()

	@classmethod
	def from_dict(cls, dict):
		s = cls(dict["ticker"], dict["datetime_run"], dict["data_endpoint"])

		s.stock_price_history = pd.read_json(dict["stock_price_history"]) if "stock_price_history" in dict else None

		option_chain_list = dict["option_chains"] if "option_chains" in dict else None
		for chain in option_chain_list:
			s.option_chains.append(OptionChain.from_dict(chain))
		return s

	def get_last_close_price(self):
		if self.stock_price_history.empty is True:
			return "Stock price history not initialized."
		return self.stock_price_history.tail(1)["Close"][0]

	def download_stock_price_data(self, start_dt = None, end_dt = None):
		self.stock_price_history = self.yfinance_ticker_object.history(start=start_dt, end=end_dt)

	def get_option_chains_merged(self, format="dataframe"):
		if self.calls == None or self.puts == None:
			return None

		if format == "dataframe":
			# Transform the option chains into one dataframe
			
			# Fix this, stupid [0] reference
			columns = self.option_chains[0].calls.columns.to_list() + ["type", "updated_time_dt", "updated_time_hours"]
			#columns = self.calls.columns.to_list()
			output_df = pd.DataFrame(columns=columns)

			# Fix this, stupid [0] reference
			calls_df = pd.DataFrame(columns=self.option_chains[0].calls.columns.to_list())
			puts_df = pd.DataFrame(columns=self.option_chains[0].puts.columns.to_list())

			for c in self.option_chains:
				update_time_temp = c.updated_time.split(" ")

				calls_df = calls_df.append(c.calls)
				calls_df["type"] = "call"
				calls_df["updated_time_dt"] = update_time_temp[0]
				calls_df["updated_time_hours"] = update_time_temp[1]

				puts_df = puts_df.append(c.puts)
				puts_df["type"] = "put"
				puts_df["updated_time_dt"] = update_time_temp[0]
				puts_df["updated_time_hours"] = update_time_temp[1]

			output_df = output_df.append(calls_df)
			output_df = output_df.append(puts_df)
			output_df["expiration_date"] = output_df.apply(lambda row: self._get_expiration_date_helper(row.name), axis=1)

			return output_df.fillna(0)

	def _get_expiration_date_helper(self, input):
		offset = len(self.ticker)
		year = "20" + input[offset+0:offset+2]
		month = input[offset+2:offset+4]
		day = input[offset+4:offset+6]
		
		return year + "-" + month + "-" + day
		#[len(self.ticker):len(self.ticker)+6]


	def update_object_based_on_data_endpoint(self):
		if (self.data_endpoint == "yfinance"):
			self.yfinance_ticker_object = yf.Ticker(self.ticker)
		else:
			self.yfinance_ticker_object = None

	def download_options_data(self):
		if self.data_endpoint == "yfinance":
			self.datetime_run = datetime.now().strftime("%Y-%m-%d %H-%M")
			try:
				# No error handling in yfinance package, hence this
				expirations = self.yfinance_ticker_object.options
				for date in expirations:
					print("Downloading options data for: " + self.ticker + " with date " + date)
					try:
						current_option = self.yfinance_ticker_object.option_chain(date)
						self.option_chains.append(OptionChain(self.yfinance_ticker_object.ticker, date, current_option.calls, current_option.puts, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
					except:
						print("Exception found with previous request, see above line.")
						self.option_chains.append(OptionChain(self.yfinance_ticker_object.ticker, date, None, None, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
			except:
				print("No option chains found for this ticker.")
	
	def as_string(self, delimiter=','):
		data = self.__dict__
		output_str = ""
		for key, value in data.items():
			# This should be implemented recursively to account for arbitrary data structures
			if (isinstance(value, list) == True):
				for it1 in value:
					for k1, v1 in value.__dict__.items():
						output_str = output_str + delimiter + str(v1)
			else:
				output_str = output_str + delimiter + str(value)
		return output_str.replace(delimiter, "", 1) # remove first delimiter
	def make_json(self):
			# TODO: Write custom encoder
			# especially problematic: isinstance(n.option_chains[0].calls, pd.DataFrame)
			return json.dumps(self, default=lambda o: o.to_json(), sort_keys=True)

def double_result_decorator(f):
	def wrapper(*args):
		value = f(*args)
		return round(value * 100, 2)

	return wrapper

class CorrelationModule:
	# The purpose of this module is calculate anything related to cross correlations of time-series
	def __init__(self):
		self.df = None
		pass

	def get_cross_correlation(self):
		# between explicit columns or all
		self.df.corr()

	def get_most_correlated_series(self):
		# between a given column and the rest of the data, which are the most correlated series
		pass

class DataObject:
	# Manages the data of the current workspace
	def __init__(self):
		pass

class Volatility:
	def __init__(self, series, series_data_field="close", type="daily"):
		# Assumes DF with index as a date or something that can be ordered
		# Input must be a dataframe with index of something to be sorted, the value with the 'series_name' 
		self.time_series_daily = series.copy(deep=True)
		self.time_series_monthly = None
		self.series_data_field = series_data_field
		self.type = type

		self._initialize()

	def _initialize(self):
		self.time_series_daily["tomorrows_close"] = self.time_series_daily[self.series_data_field].shift(-1)
		
		self.time_series_daily["change"] = self.time_series_daily.apply(lambda row: (row["tomorrows_close"] - row[self.series_data_field]) / row[self.series_data_field], axis=1 )
		self.time_series_daily["change_log"] = self.time_series_daily.apply(lambda row: math.log(row["tomorrows_close"] / row[self.series_data_field]), axis=1 )
		
		self.time_series_daily.drop(["tomorrows_close"], inplace=True, axis=1)
		self.time_series_daily.dropna(inplace=True)
		
		if type is "daily":
			temp_monthly_df = self.time_series_daily.copy(deep=True) 
			temp_monthly_df["is_month_end"] = temp_monthly_df.index.is_month_end
			self.time_series_monthly = temp_monthly_df[temp_monthly_df.is_month_end == True]
			self.time_series_monthly["next_months_close"] = self.time_series_monthly.shift(-1)
			self.time_series_monthly["monthly_change"] = self.time_series_monthly.apply(lambda row: (row["next_months_close"] - row[self.series_data_field]) / row[self.series_data_field], axis=1 )
			self.time_series_monthly.drop(["monthly_change"], inplace=True, axis=1)
			self.time_series_monthly.dropna(inplace=True)
	
	@double_result_decorator
	def get_annualized(self, last_days=None):
		# Assumes that price_history is by day
		if last_days == None:
			return self.time_series_daily["change"].std() * math.sqrt(252)
		
		# Return the annualized volatility based on last_days number of days. 
		# Assumes the data is sorted ascending
		return self.time_series_daily.tail(last_days)["change"].std() * math.sqrt(252)
	
	@double_result_decorator	
	def get_daily(self, last_days=None):
		if last_days == None:
			return self.time_series_daily["change"].std()
		
		return self.time_series_daily.tail(last_days)["change"].std()

	@double_result_decorator
	def get_monthly(self, last_periods=None, daily_scaled=True):
		if daily_scaled == True:
			# Assumes that price_history is by day
			# Assumes 21 trading days

			if last_periods == None:
				return self.time_series_daily["change"].std() * math.sqrt(21)
		
			# Return the annualized volatility based on last_days number of days. 
			# Assumes the data is sorted ascending
			return self.time_series_daily.tail(last_periods)["change"].std() * math.sqrt(21)
		
		if last_periods == None:
			return self.time_series_monthly["monthly_change"].std()
		return self.time_series_monthly["monthly_change"].tail(last_periods).std()
	
	@double_result_decorator
	def get_weekly(self):
		# Assumes that price_history is by day
		# Assumes 5 trading days

		if last_days == None:
			return self.time_series_daily["change"].std() * math.sqrt(5)
		
		# Return the annualized volatility based on last_days number of days. 
		# Assumes the data is sorted ascending
		return self.time_series_daily.tail(last_days)["change"].std() * math.sqrt(5)

	@double_result_decorator
	def	get_realized_volatility(self, period_type, periods, roll_period=0):
		# daily, weekly, monthly, yearly
		if period_type == 'daily':
			return self.time_series_daily.tail(periods)["change"].std()

		if period_type == 'weekly':
			return None

		if period_type == 'monthly':
			return self.time_series_monthly.tail(periods)["monthly_change"].std()
	
		if period_type == 'rolling':
			return self.time_series_daily.rolling(roll_period).std()

	def plot_realized_volatility(self, period_type, roll_periods):
		# params: 
		#	'period_type' : string, 'rolling'
		#	'roll_periods' : list of numbers, '[5, 20, 60]'
		if period_type == 'rolling':
			for p in roll_periods:
				plt.plot(self.time_series_daily[self.series_data_field].rolling(p).std(), label=str(p))
			plt.legend()
			plt.show()

	def get_mean(self):
		pass

	def get_moving_average(self):
		pass

	def plot_acf(self):
		plot_acf(self.time_series_daily["change"])
		plt.show()

	def plot_histogram(self, bins=250, log=False):
		if log is False:
			self.time_series_daily.hist(column="change", bins=bins)
		else:
			self.time_series_daily.hist(column="change_log", bins=bins)
		plt.show()

	def get_return_descriptive_statistics(self, number_of_periods=None):
		# mean, mode, std, kurtosis, skewness
		print("Mean: " + str(self.time_series_daily["change"].mean()))
		print("Median: " + str(self.time_series_daily["change"].median()))
		print("Mode: " + str(self.time_series_daily["change"].mode()))
		print("Std: " + str(self.time_series_daily["change"].std()))
		print("Kurtosis: " + str(self.time_series_daily["change"].kurt()))
		print("Skew: " + str(self.time_series_daily["change"].skew()))
		

	def plot(self):
		pass