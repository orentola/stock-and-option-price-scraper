import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import copy
import QuantLib as ql 

import pandas as pd
import json

from stockmodule import Stock
from stockmodule import Volatility
import stockmodule

import seaborn as sns

from scipy import stats
from scipy.stats import shapiro
from scipy.stats import norm

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from matplotlib import pyplot
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot

from mpl_toolkits.mplot3d import Axes3D

import datetime

import math
import numpy as np

import optionstrategypricingmodule
import multiprocessing as mp

#STOCK_DATA_PATH = "C:\\Users\\orent\\Documents\\StockDataDownloader\\2020-09-11_21_44_one_time_run\\data.json"
STOCK_DATA_PATH = "/home/orentola/stock-and-option-price-scraper/stock-and-option-price-scraper/stock_data/2020-09-13_04_39_one_time_run/data.json"

underlying_price_time_series_value_list = []
option_time_series_value_list = []
total_profit_loss = []

def simulator_price_collector_callback(result):
	# result[0] => underlying price time series
	# result[1] => (profit_loss, leg_values)

	global underlying_price_time_series_value_list
	global option_time_series_value_list
	global total_profit_loss

	underlying_price_time_series_value_list.append(result[0])
	
	option_time_series_value_list.append(result[1][1])	
	total_profit_loss.append(result[1][0]["total_profit_loss"])

	#print("Completed iteration: " + str(option_time_series_value.shape[0]))

def main():
	scenario_data = []
	#strikes = [190]
	strikes = [x for x in np.arange(180, 205, 2.5)]
	#spread_widths = [2.5, 5.0, 10.0]
	spread_widths = [2.5, 5, 10]
	stock_price_dict = {}
	option_price_dict = {}

	# TODO: NEXT STEP, RUN THIS SHIT AND SEE WHAT HAPPENS

	for strike in strikes:
		for width in spread_widths:
			ticker = "MSFT"
			strike_price = strike
			number_of_simulations = 1000
			dividend_rate = 0.0095
			RISK_FREE_RATE = 0.008

			manual_input_volatility = True

			start_date = "2020-05-01" # YYYY-MM-DD
			maturity_date = "2020-06-10" # YYYY-MM-DD

			optionLegsDict = {}

			optionLegDict1 = {}
			optionLegDict1["name"] = "leg_name2"
			optionLegDict1["ticker"] = ticker
			optionLegDict1["dividend_rate"] = dividend_rate
			optionLegDict1["option"] = "Put"
			optionLegDict1["position_type"] = "short"
			optionLegDict1["volatility"] = 0.387
			optionLegDict1["strike_price"] = strike_price
			optionLegDict1["maturity_date"] = maturity_date
			optionLegDict1["risk_free_rate"] = RISK_FREE_RATE
			optionLegDict1["start_date"] = start_date
			optionLegsDict[optionLegDict1["name"]] = optionLegDict1

			optionLegDict2 = {}
			optionLegDict2["name"] = "leg_name1"
			optionLegDict2["ticker"] = ticker
			optionLegDict2["dividend_rate"] = dividend_rate
			optionLegDict2["option"] = "Put"
			optionLegDict2["position_type"] = "long" 
			optionLegDict2["volatility"] = 0.387 + 0.0125
			optionLegDict2["strike_price"] = strike_price - width
			optionLegDict2["maturity_date"] = maturity_date
			optionLegDict2["risk_free_rate"] = RISK_FREE_RATE
			optionLegDict2["start_date"] = start_date
			optionLegsDict[optionLegDict2["name"]] = optionLegDict2
		
			underlying_price_time_series_value_list.clear()
			option_time_series_value_list.clear()
			total_profit_loss.clear()

			date_difference = (datetime.datetime.strptime(maturity_date, "%Y-%m-%d") - datetime.datetime.strptime(start_date, "%Y-%m-%d")).days
			# Sample for the duration
			samples = date_difference

			s = optionstrategypricingmodule.StockPriceService(STOCK_DATA_PATH)
			# TODO SUPPORT FOR MULTIPLE TICKERS IN KDE
			s.make_kde(ticker)
			s.get_sample_of_current_kde(10000)
			#s.fit_normal_dist(ticker)
			#s.plot_histogram(ticker, True, True)

			current_spot_price = s.get_last_close_price(ticker)

			if manual_input_volatility is False:
				current_volatility_daily = s.get_volatility(ticker, 30) / 100
				current_volatility = current_volatility_daily * math.sqrt(365) # THIS MAY NEED TO BE 252
			
				for k, v in optionLegsDict.items():
					v["volatility"] = current_volatility
		

			mp_start_dt = datetime.datetime.now()
			pool = mp.Pool(mp.cpu_count()-1)

			# For testing and debugging purposes
			#test = optionstrategypricingmodule.getSimulatedOptionPriceForOneIteration(ticker, s, samples, optionLegsDict, current_spot_price)

			for i in range(0, number_of_simulations):
				pool.apply_async(optionstrategypricingmodule.getSimulatedOptionPriceForOneIteration, args=(ticker, s, samples, optionLegsDict, current_spot_price), callback=simulator_price_collector_callback)

			pool.close()
			pool.join()
			mp_end_dt = datetime.datetime.now()

			underlying_price_time_series_value = pd.DataFrame(underlying_price_time_series_value_list)
			option_time_series_value = pd.DataFrame(total_profit_loss)

			#option_time_series_value_list

			option_time_series_value.describe()
			break_even_threshold_price = 0.0 # Sum of all profits and losses of individual legs, breakeven p/l at 0.0
			#option_time_series_value.iloc[0,0] # Price at time=0
			percentile_breakeven_at_end = stats.percentileofscore(option_time_series_value[underlying_price_time_series_value.shape[1]-1], break_even_threshold_price)
	
			expected_value_at_end = (option_time_series_value[underlying_price_time_series_value.shape[1]-1] * (1/number_of_simulations)).sum()
			expected_value_daily = (option_time_series_value * (1/number_of_simulations)).sum()
			expected_value_daily.name = "Expected"
			percentile_breakeven_daily = stats.percentileofscore(expected_value_daily, break_even_threshold_price)
		
			underlying_price_time_series_value.describe()
			stock_price_estimate_lower_bound = underlying_price_time_series_value.quantile(0.05)
			stock_price_estimate_median = underlying_price_time_series_value.quantile(0.50)
			stock_price_estimate_upper_bound = underlying_price_time_series_value.quantile(0.95)
		
			quantiles = [0.05, 0.20, 0.50, 0.80, 0.95]
			option_price_quantiles = option_time_series_value.quantile(quantiles)
			option_price_quantiles = option_price_quantiles.append(expected_value_daily)
			break_even_threshold_price_series = pd.Series([break_even_threshold_price for i in range(0, date_difference)], name="Breakeven")
			option_price_quantiles = option_price_quantiles.append(break_even_threshold_price_series)

			#for index, row in option_price_quantiles.iterrows():
			#	plt.plot(row, label=index)
			#plt.xlabel("Days")
			#plt.ylabel("Profit Loss $")
			#plt.legend()
			#plt.show()

			#option_price_estimate_lower_bound = option_time_series_value.quantile(0.05)
			#option_price_estimate_expected_value = expected_value_daily
			#option_price_estimate_median = option_time_series_value.quantile(0.50)
			#option_price_estimate_upper_bound = option_time_series_value.quantile(0.95)
			#option_price_estimate_upper_bound_75 = option_time_series_value.quantile(0.75)
			#option_price_estimate_upper_bound_85 = option_time_series_value.quantile(0.85)

			stock_price_estimate_df = pd.DataFrame(data={"Lower Bound" : stock_price_estimate_lower_bound, "Upper Bound" : stock_price_estimate_upper_bound, "Median" : stock_price_estimate_median})

			#plt.plot(stock_price_estimate_df['Lower Bound'], label='Lower Bound')
			#plt.plot(stock_price_estimate_df['Median'], label='Median')
			#plt.plot(stock_price_estimate_df['Upper Bound'], label='Upper Bound')
			#plt.legend()
			#plt.show()
		
			stock_price_dict[str(strike) + "_" + str(width)] = copy.deepcopy(stock_price_estimate_df)

		
			#option_price_estimate_df = pd.DataFrame(data={"Upper Bound 85%": option_price_estimate_upper_bound_85, "Upper Bound 75%": option_price_estimate_upper_bound_75, "Breakeven" : break_even_threshold_price_series, "Lower Bound 5%" : option_price_estimate_lower_bound, "Upper Bound 95%": option_price_estimate_upper_bound, "Median 50%" : option_price_estimate_median, "Expected" : option_price_estimate_expected_value})

			#plt.plot(option_price_estimate_df['Lower Bound 5%'], label='Lower Bound 5%')
			#plt.plot(option_price_estimate_df['Median 50%'], label='Median')
			#plt.plot(option_price_estimate_df['Upper Bound 95%'], label='Upper Bound 95%')
			#plt.plot(option_price_estimate_df['Expected'], label='Expected')
			#plt.plot(option_price_estimate_df['Breakeven'], label='Breakeven')
			#plt.plot(option_price_estimate_df['Upper Bound 75%'], label='Upper Bound 75%')
			#plt.plot(option_price_estimate_df['Upper Bound 85%'], label='Upper Bound 85%')
			#plt.legend()
			#plt.show()

			option_price_dict[str(strike) + "_" + str(width)] = copy.deepcopy(option_price_quantiles)

			# If expected simulated value is greater than the price at time 0, it is smarter to buy the call option vs. sell

	#plt.plot(option_price_dict[185]['Expected'], label='Expected 185')
	#plt.plot(option_price_dict[190]['Expected'], label='Expected 190')
	#plt.plot(option_price_dict[195]['Expected'], label='Expected 195')
	#plt.plot(option_price_dict[200]['Expected'], label='Expected 200')
	#plt.plot(option_price_dict[205]['Expected'], label='Expected 205')
	#plt.plot(option_price_dict[185]['Breakeven'], label='Breakeven 185')
	#plt.plot(option_price_dict[190]['Breakeven'], label='Breakeven 190')
	#plt.plot(option_price_dict[195]['Breakeven'], label='Breakeven 195')
	#plt.plot(option_price_dict[200]['Breakeven'], label='Breakeven 200')
	#plt.plot(option_price_dict[205]['Breakeven'], label='Breakeven 205')
	#plt.legend()
	#plt.show()

	output_dict = {}

	for k in option_price_dict.keys():
		output_dict[k] = option_price_dict[k].to_json()

	with open("C:\\Users\\orent\\Documents\\testest.json", "w") as f:
		json.dump(output_dict, f)

	with open("C:\\Users\\orent\\Documents\\dump.json", "r") as f:
		data = json.load(f)
	#plt.plot(option_price_dict[185]['Upper Bound 85%'], label='Upper Bound 185 85%')
	#plt.plot(option_price_dict[190]['Upper Bound 85%'], label='Upper Bound 190 85%')
	#plt.plot(option_price_dict[195]['Upper Bound 85%'], label='Upper Bound 195 85%')
	#plt.plot(option_price_dict[200]['Upper Bound 85%'], label='Upper Bound 200 85%')
	#plt.plot(option_price_dict[205]['Upper Bound 85%'], label='Upper Bound 205 85%')
	#plt.plot(option_price_dict[185]['Breakeven'], label='Breakeven 185')
	#plt.plot(option_price_dict[190]['Breakeven'], label='Breakeven 190')
	#plt.plot(option_price_dict[195]['Breakeven'], label='Breakeven 195')
	#plt.plot(option_price_dict[200]['Breakeven'], label='Breakeven 200')
	#plt.plot(option_price_dict[205]['Breakeven'], label='Breakeven 205')
	#plt.legend()
	#plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")

	#for k in option_price_dict.keys():
	for k in data.keys():
		#eb = (option_price_dict[k].loc['Expected', :])[samples-1] 
		#lb = (option_price_dict[k].loc[0.05, :])[samples-1]
		#ub = (option_price_dict[k].loc[0.95, :])[samples-1]
		#plt.scatter(option_price_dict[k].loc['Expected', :], option_price_dict[k].loc[0.05, :])
		#ax.scatter(option_price_dict[k].loc['Expected', :], option_price_dict[k].loc[0.05, :], option_price_dict[k].loc['Expected', :].index.to_list(), label=k)
		#ax.scatter(pd.read_json(data[k]).loc['Expected', :], pd.read_json(data[k]).loc["0.05", :], pd.read_json(data[k]).loc['Expected', :].index.to_list(), label=k)
		current_df = pd.read_json(data[k])
		ax.scatter(current_df.loc['Expected', :], current_df.loc["0.05", :], (current_df.loc['Expected', :] / current_df.loc["0.05", :]).apply(abs), label=k)
	
	ax.set_xlabel("Expected Value At Time")
	ax.set_ylabel("Tail Risk 95%")
	#ax.set_zlabel("Time to expiration")
	ax.set_zlabel("Expected value vs. tail risk, higher better")
	ax.legend(loc="upper right")

	plt.show()
	
	for k in data.keys():
		current_df = pd.read_json(data[k])
		#plt.scatter(current_df.loc['Expected', :], (current_df.loc['Expected', :] / current_df.loc["0.05", :]).apply(abs), label=k)
		plt.scatter(current_df.loc['Expected', :].mean(), (current_df.loc['Expected', :] / current_df.loc["0.05", :]).apply(abs).mean(), label=k)
	plt.legend()
	plt.xlabel("Expected")
	plt.ylabel("Expected vs. Tail Risk, higher better for E>0")
	plt.show()
	

	# Calculate expected upside for different levels
	upside = {}
	x = []
	lb_list = []
	ub_list = []
	strike_list_for_plotting = []
	spread_width = []

	for k in option_price_dict.keys():
		#eb = (option_price_dict[k].loc['Expected', :])[samples-1] 
		#lb = (option_price_dict[k].loc[0.05, :])[samples-1]
		#ub = (option_price_dict[k].loc[0.95, :])[samples-1]

		eb = (option_price_dict[k].loc['Expected', :]) 
		lb = (option_price_dict[k].loc[0.05, :])
		ub = (option_price_dict[k].loc[0.95, :])
		
		strike_list_for_plotting.append(float(k.split("_")[0])) 

		spread_width.append(float(k.split("_")[1])) 

		#upside[s] = (eb, ub)
		x.append(eb)
		#x.append(eb)
		ub_list.append(ub)
		lb_list.append(lb)

	#plt.plot(x, lb_list)
	#plt.plot(x, ub_list)
	keys = [x for x in option_price_dict.keys()]
	plt.scatter(x, lb_list)
	for i in range(0, len(keys)):
		plt.annotate(keys[i], (x[i], lb_list[i]))
	plt.show()

	#for i, txt in enumerate(strike_list_for_plotting):
	#	ax.annotate(txt, (x[i], lb_list[i]))
	output_data = {}
	output_data["x"] = x
	output_data["lb_list"] = lb_list
	output_data["ub_list"] = ub_list
	output_data["spread_width"] = spread_width
	output_df = pd.DataFrame(data=output_data)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")

	ax.scatter(x, lb_list, spread_width, c='r', marker='o')
	ax.set_xlabel("Expected Value At End")
	ax.set_ylabel("Tail Risk 95%")
	ax.set_zlabel("Spread Width")

	plt.show()

	#plt.scatter(x, ub_list)


	plt.xlabel("Expected")
	plt.ylabel("Upper/lower bound")
	plt.show()