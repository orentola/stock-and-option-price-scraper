import yfinance as yf
from datetime import datetime
import pandas as pd
import os
import copy


TICKERS_LIST = 'tickers.txt'
TICKER_LIST_NAME = 'CUSTOM LIST'
#TICKER_LIST_NAME = 'DEBUG'

def main():
	d = Downloader(TICKER_LIST_NAME)
	d.initialize()
	d.download_options_data()

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
	def __init__(self, ticker, date, calls, puts, updated_time):
		self.ticker = ticker
		self.date = date
		self.calls = calls
		self.puts = puts
		self.updated_time = updated_time

class Stock:
	def __init__(self, ticker):
		self.ticker = ticker
		self.stock_price_data = None
		self.option_chains = []
		self.yfinance_ticker_object = None
		self.data_endpoint = "yfinance"

		self.update_object_based_on_data_endpoint()

	def update_object_based_on_data_endpoint(self):
		if (self.data_endpoint == "yfinance"):
			self.yfinance_ticker_object = yf.Ticker(self.ticker)
		else:
			self.yfinance_ticker_object = None

	def download_options_data(self):
		if self.data_endpoint == "yfinance":
			for date in self.yfinance_ticker_object.options:
				print("Downloading options data for: " + self.ticker + " with date " + date)
				try:
					current_option = self.yfinance_ticker_object.option_chain(date)
					self.option_chains.append(OptionChain(self.yfinance_ticker_object.ticker, date, current_option.calls, current_option.puts, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
				except:
					print("Exception found with previous request, see above line.")
					self.option_chains.append(OptionChain(self.yfinance_ticker_object.ticker, date, None, None, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
	
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

	def to_json(self):
		# TODO: Write custom encoder
		# especially problematic: isinstance(n.option_chains[0].calls, pd.DataFrame)
		return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

s = yf.Ticker("MSFT")
s.info

hist = s.history(period='max')

s.options

for date in s.options:
	cur_opt = s.option_chain(date)
	cur_opt.calls
	cur_opt.puts