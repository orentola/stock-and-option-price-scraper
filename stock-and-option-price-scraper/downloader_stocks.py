import yfinance as yf
import pandas as pd
import os
import copy
import pytz
import json

from json import JSONEncoder
from datetime import datetime
from time import sleep
from pathlib import Path

from stockmodule import Stock
from stockmodule import Downloader
from stockmodule import OptionChain
from stockmodule import StockEncoder
from stockmodule import DownloaderEncoder

TICKERS_LIST = 'tickers.txt'
#TICKER_LIST_NAME = 'SP500+SOFTWARE_TECH'
TICKER_LIST_NAME = 'CUSTOM LIST'
#TICKER_LIST_NAME = 'TEST'
DATA_PATH = "C:\\Users\\orent\\Documents\\StockDataDownloader\\"

# Fix this implementation for running this once, 
# right now implemented in a bad way
RUN_ONCE = True

def main():
	print("Starting the run with ticker list: " + TICKER_LIST_NAME)
	d = Downloader(TICKER_LIST_NAME)

	start_dt = "2000-01-01"
	end_dt = datetime.strftime(datetime.today(), "%Y-%m-%d")

	print("Starting the loop for checking time and running all the steps...")
	while True:
		if RUN_ONCE is False:
			if (datetime.today().weekday() not in DAYS_WHEN_RUN):
				sleep_minutes = (24 - datetime.now().hour) * 60
				print("It seems that this isn't supposed to run today. Sleeping for " + str(sleep_minutes) )
				sleep(sleep_minutes * 60)
			else:
				print("It seems today is a weekday, let's run!")

				run_name = os.path.join(datetime.today().strftime("%Y-%m-%d_%H_%M"))
		else:
			run_name = os.path.join(datetime.today().strftime("%Y-%m-%d_%H_%M_one_time_run"))
		
		print("Initializing downloader.")
		d.initialize()
		print("Initialization complete.")

		print(datetime.now().strftime("%Y-%m-%d %H:%M") + ": Starting to download stock data.")
		d.download_stock_data(start_dt, end_dt)
		print(datetime.now().strftime("%Y-%m-%d %H:%M") + ": Stock data download completed.")

		print("Dumping Downloader object.")
		json_dump = json.dumps(d, cls=DownloaderEncoder)
		print("Downloader object successfully dumped.")

		print("Writing the JSON dump to disk.")

		output_file_folder = os.path.join(DATA_PATH, run_name)
		Path(output_file_folder).mkdir(parents=True, exist_ok=True)

		with open( os.path.join(output_file_folder, "data.json"), "w+") as f:
			f.write(json_dump)
		print("JSON dumped successfully to the disk.")

		if RUN_ONCE is True:
			break



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
