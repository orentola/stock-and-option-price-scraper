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

import datetime

import math
import numpy as np

import optionstrategypricingmodule
import multiprocessing as mp

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
	strikes = [185]
	#strikes = [x for x in range(180, 205, 2.5)]
	stock_price_dict = {}
	option_price_dict = {}

	# TODO: NEXT STEP, RUN THIS SHIT AND SEE WHAT HAPPENS

	for strike in strikes:
		ticker = "MSFT"
		strike_price = strike
		number_of_simulations = 100
		dividend_rate = 0.0095
		RISK_FREE_RATE = 0.008

		manual_input_volatility = False

		start_date = "2020-05-01" # YYYY-MM-DD
		maturity_date = "2020-06-10" # YYYY-MM-DD

		optionLegsDict = {}

		optionLegDict1 = {}
		optionLegDict1["name"] = "leg_name2"
		optionLegDict1["ticker"] = ticker
		optionLegDict1["dividend_rate"] = dividend_rate
		optionLegDict1["option"] = "Put"
		optionLegDict1["position_type"] = "short"
		optionLegDict1["volatility"] = ""
		optionLegDict1["strike_price"] = strike_price
		optionLegDict1["maturity_date"] = maturity_date
		optionLegDict1["risk_free_rate"] = RISK_FREE_RATE
		optionLegDict1["start_date"] = start_date

		#optionLegDict2 = {}
		#optionLegDict2["name"] = "leg_name1"
		#optionLegDict2["ticker"] = ticker
		#optionLegDict2["dividend_rate"] = dividend_rate
		#optionLegDict2["option"] = "Put"
		#optionLegDict2["position_type"] = "long" 
		#optionLegDict2["volatility"] = ""
		#optionLegDict2["strike_price"] = strike_price - 5
		#optionLegDict2["maturity_date"] = maturity_date
		#optionLegDict2["risk_free_rate"] = RISK_FREE_RATE
		#optionLegDict2["start_date"] = start_date

		optionLegsDict[optionLegDict1["name"]] = optionLegDict1
		#optionLegsDict[optionLegDict2["name"]] = optionLegDict2

		underlying_price_time_series_value_list.clear()
		option_time_series_value_list.clear()
		total_profit_loss.clear()

		date_difference = (datetime.datetime.strptime(maturity_date, "%Y-%m-%d") - datetime.datetime.strptime(start_date, "%Y-%m-%d")).days
		# Sample for the duration
		samples = date_difference

		s = optionstrategypricingmodule.StockPriceService()
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
		pool = mp.Pool(mp.cpu_count())

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
		break_even_threshold_price = option_time_series_value.iloc[0,0] # Price at time=0
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

		# TODO: Breakeven price is not correct 
		for index, row in option_price_quantiles.iterrows():
			plt.plot(row, label=index)
		plt.legend()
		plt.show()
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
		
		stock_price_dict[strike_price] = copy.deepcopy(stock_price_estimate_df)

		
		#option_price_estimate_df = pd.DataFrame(data={"Upper Bound 85%": option_price_estimate_upper_bound_85, "Upper Bound 75%": option_price_estimate_upper_bound_75, "Breakeven" : break_even_threshold_price_series, "Lower Bound 5%" : option_price_estimate_lower_bound, "Upper Bound 95%": option_price_estimate_upper_bound, "Median 50%" : option_price_estimate_median, "Expected" : option_price_estimate_expected_value})

		plt.plot(option_price_estimate_df['Lower Bound 5%'], label='Lower Bound 5%')
		plt.plot(option_price_estimate_df['Median 50%'], label='Median')
		plt.plot(option_price_estimate_df['Upper Bound 95%'], label='Upper Bound 95%')
		plt.plot(option_price_estimate_df['Expected'], label='Expected')
		plt.plot(option_price_estimate_df['Breakeven'], label='Breakeven')
		plt.plot(option_price_estimate_df['Upper Bound 75%'], label='Upper Bound 75%')
		plt.plot(option_price_estimate_df['Upper Bound 85%'], label='Upper Bound 85%')
		plt.legend()
		plt.show()

		option_price_dict[strike_price] = copy.deepcopy(option_price_estimate_df)

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

	# Calculate expected upside for different levels
	upside = {}
	x = []
	y = []
	for s in strikes:
		eb = (option_price_dict[s]['Expected'][samples-1] - option_price_dict[s]['Breakeven'][samples-1]) / option_price_dict[s]['Breakeven'][samples-1]
		ub = (option_price_dict[s]['Upper Bound 85%'][samples-1] - option_price_dict[s]['Breakeven'][samples-1]) / option_price_dict[s]['Breakeven'][samples-1]
		upside[s] = (eb, ub)
		x.append(eb)
		y.append(ub)

	plt.plot(x, y)
	plt.xlabel("Expected P/L %")
	plt.ylabel("85% P/L %")
	plt.show()