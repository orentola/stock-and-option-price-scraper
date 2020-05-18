import yfinance as yf
import pandas as pd
import os
import copy
import pytz

from json import JSONEncoder
from datetime import datetime
from time import sleep
from pathlib import Path

TICKERS_LIST = 'tickers.txt'
TICKER_LIST_NAME = 'CUSTOM LIST'
#TICKER_LIST_NAME = 'DEBUG'
DATA_PATH = "C:\\Users\\orent\\Documents\\OptionsDataDownloader\\"

HOURS_WHEN_RUN = [7, 8, 9, 10, 11, 12, 13, 16] #PST times
DAYS_WHEN_RUN = [0, 1, 2, 3, 4] # Weekdays to run

def main():
	print("Starting the run with ticker list: " + TICKER_LIST_NAME)
	d = Downloader(TICKER_LIST_NAME)

	hour_queue = []

	print("Starting the loop for checking time and running all the steps...")
	while True:
		if (datetime.today().weekday() not in DAYS_WHEN_RUN):
			sleep_minutes = (24 - datetime.now().hour) * 60
			print("It seems that this isn't supposed to run today. Sleeping for " + str(sleep_minutes) )
			sleep(sleep_minutes * 60)
		else:
			print("It seems today is a weekday, let's run!")
			if len(hour_queue) == 0:
				print("It seems that the queue for hours is not initialized. Let's initialize.")
				hour_queue = copy.deepcopy(HOURS_WHEN_RUN)

				# Make sure we don't get stuck here if the script is launched at 
				# different time of day

				while (hour_queue[0] < datetime.now().hour):
					hour_queue.pop(0)

					if len(hour_queue) == 0:
						# Seems like we are starting the script after the trading day
						# waiting until tomorrow to try again. If you want to run just the 
						# last day, you'll need to run this separately. 

						# Subtracting 60 to ensure we're starting before the first hour to run

						sleep_time_minutes = (datetime.now().hour - HOURS_WHEN_RUN[0]) * 60 - 60
						print("Looks like we're initiating the script quite late, will sleep until next morning: " + str(sleep_time_minutes))
						sleep(sleep_time_minutes * 60)
						continue
			
			# We're good to run the next hour in the queue
			print("Next run in line: " + str(hour_queue[0]) + ". Current time: " + datetime.now().hour + ":" + datetime.now().minute)
			sleep_minutes = 60 - datetime.now().minute
			sleep_hours_minutes = (hour_queue[0] - datetime.now().hour) * 60

			print("Sleeping for: " + str(sleep_minutes + sleep_hours_minutes))
			sleep((sleep_minutes + sleep_hours_minutes) * 60)
			
			run_name = os.path.join(datetime.today().strftime("%Y-%m-%d"), str(hour_queue[0]))

			print("Initializing downloader.")
			d.initialize()
			print("Initialization complete.")

			print(datetime.now().strftime("%Y-%m-%d %H:%M") + ": Starting to download options data.")
			d.download_options_data()
			print(datetime.now().strftime("%Y-%m-%d %H:%M") + ": Options data download completed.")

			print("Dumping Downloader object.")
			json_dump = json.dumps(d, cls=DownloaderEncoder)
			print("Downloader object successfully dumped.")

			print("Writing the JSON dump to disk.")

			output_file_folder = os.path.join(DATA_PATH, run_name)
			Path(output_file_folder).mkdir(parents=True, exist_ok=True)

			with open( os.path.join(output_file_folder, "data.json"), "w+") as f:
				f.write(json_dump)
			print("JSON dumped successfully to the disk.")

			hour_queue.pop(0)

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
		calls = pd.read_json(dict["calls"]) if dict["calls"] is not None else None
		puts = pd.read_json(dict["puts"]) if dict["puts"] is not None else None
		updated_time = dict["updated_time"]
		return OptionChain(ticker, date, calls, puts, updated_time)

class Stock:
	def __init__(self, ticker, datetime_run = None, data_endpoint = "yfinance"):
		self.ticker = ticker
		self.stock_price_data = None
		self.option_chains = []
		self.yfinance_ticker_object = None
		self.data_endpoint = "yfinance"
		self.datetime_run = None

		self.update_object_based_on_data_endpoint()

	@classmethod
	def from_dict(cls, dict):
		s = cls(dict["ticker"], dict["datetime_run"], dict["data_endpoint"])
		option_chain_list = dict["option_chains"]
		for chain in option_chain_list:
			s.option_chains.append(OptionChain.from_dict(chain))
		return s

	def update_object_based_on_data_endpoint(self):
		if (self.data_endpoint == "yfinance"):
			self.yfinance_ticker_object = yf.Ticker(self.ticker)
		else:
			self.yfinance_ticker_object = None

	def download_options_data(self):
		if self.data_endpoint == "yfinance":
			self.datetime_run = datetime.now().strftime("%Y-%m-%d %H-%M")
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
	def make_json(self):
			# TODO: Write custom encoder
			# especially problematic: isinstance(n.option_chains[0].calls, pd.DataFrame)
			return json.dumps(self, default=lambda o: o.to_json(), sort_keys=True)

#############################
# Helper functions

def stock_as_dict(obj):
	output_dict = {}
	output_dict["ticker"] = obj.ticker
	output_dict["data_endpoint"] = obj.data_endpoint
	output_dict["datetime_run"] = obj.datetime_run
			
	option_chain_list = []
	for it in obj.option_chains:
		option_chain_list.append(it.to_json())

	output_dict["option_chains"] = option_chain_list 
	return output_dict

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

## For Debugging purposes
#t = json.dumps(d, cls=DownloaderEncoder)
#t = json.dumps(s, cls=StockEncoder)

#output_path = os.path.join(os.getcwd(), "test_json_all.json")
#with open(output_path, "w") as f:
#	f.write(t)

#with open(output_path, "r") as f:
#	data = json.load(f)

###################################

main()
