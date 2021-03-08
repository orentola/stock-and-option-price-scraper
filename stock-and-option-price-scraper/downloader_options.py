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
#TICKER_LIST_NAME = 'DEBUG'
DATA_PATH = "C:\\Users\\orent\\Documents\\OptionsDataDownloader\\"

# Fix this implementation for running this once, 
# right now implemented in a bad way
RUN_ONCE = True

HOURS_WHEN_RUN = [7, 8, 9, 10, 11, 12, 13, 16] #PST times
DAYS_WHEN_RUN = [0, 1, 2, 3, 4] # Weekdays to run

def main():
	print("Starting the run with ticker list: " + TICKER_LIST_NAME)
	d = Downloader(TICKER_LIST_NAME)

	hour_queue = []

	print("Starting the loop for checking time and running all the steps...")
	while True:
		if RUN_ONCE is False:
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
							hour_queue.append(HOURS_WHEN_RUN[0])
							break
			
				# We're good to run the next hour in the queue
				print("Next run in line: " + str(hour_queue[0]) + ". Current time: " + str(datetime.now().hour) + ":" + str(datetime.now().minute))
				sleep_minutes = 60 - datetime.now().minute
				sleep_hours_minutes = (hour_queue[0] - datetime.now().hour) * 60

				print("Sleeping for minutes: " + str(sleep_minutes + sleep_hours_minutes))
				time_to_sleep = (sleep_minutes + sleep_hours_minutes) * 60 if sleep_minutes + sleep_hours_minutes > 0 else 10
				sleep(time_to_sleep)
			
				run_name = os.path.join(datetime.today().strftime("%Y-%m-%d"), str(hour_queue[0]))
		else:
			run_name = os.path.join(datetime.today().strftime("%Y-%m-%d"), "one_time_run_" + str(datetime.now().hour))
		
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

		if RUN_ONCE is not True:
			hour_queue.pop(0)

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
