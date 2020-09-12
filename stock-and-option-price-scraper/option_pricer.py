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

underlying_price_time_series_value = pd.DataFrame()
option_time_series_value = pd.DataFrame()

def simulator_price_collector_callback(result):
	global underlying_price_time_series_value
	global option_time_series_value

	underlying_price_time_series_value = underlying_price_time_series_value.append(result[0])
	option_time_series_value = option_time_series_value.append(result[1])

	#print("Completed iteration: " + str(option_time_series_value.shape[0]))

def main():
	ticker = "MSFT"
	strike_price = 190
	number_of_simulations = 1000
	dividend_rate = 0.00
	risk_free_rate = 0.008
	option_type = "Put"
	position_type = "Short"

	start_date = "2020-05-01" # YYYY-MM-DD
	maturity_date = "2020-06-10" # YYYY-MM-DD

	date_difference = (datetime.datetime.strptime(maturity_date, "%Y-%m-%d") - datetime.datetime.strptime(start_date, "%Y-%m-%d")).days
	# Sample for the duration
	samples = date_difference

	s = optionstrategypricingmodule.StockPriceService()
	s.make_kde(ticker)
	s.get_sample_of_current_kde(10000)
	s.fit_normal_dist(ticker)
	s.plot_histogram(ticker, True, True)

	current_spot_price = s.get_last_close_price(ticker)
	current_volatility_daily = s.get_volatility(ticker, 30) / 100
	current_volatility = current_volatility_daily * math.sqrt(365) # THIS MAY NEED TO BE 252
	#current_volatility = 0.93
	
	#option_time_series_value = pd.DataFrame()
	#underlying_price_time_series_value = pd.DataFrame()

	optionLegDict = {}
	optionLegDict["ticker"] = ticker
	optionLegDict["dividend_rate"] = dividend_rate
	optionLegDict["option"] = option_type
	optionLegDict["position_type"] = position_type
	optionLegDict["volatility"] = current_volatility
	optionLegDict["strike_price"] = strike_price
	optionLegDict["maturity_date"] = maturity_date
	optionLegDict["risk_free_rate"] = risk_free_rate
	optionLegDict["start_date"] = start_date

	#option_time_series_value = []

	mp_start_dt = datetime.datetime.now()
	pool = mp.Pool(mp.cpu_count())

	for i in range(0, number_of_simulations):
		#option_time_series_value = optionstrategypricingmodule.getSimulatedOptionPriceForOneIteration(ticker, s, samples, optionLegDict, current_spot_price)
		pool.apply_async(optionstrategypricingmodule.getSimulatedOptionPriceForOneIteration, args=(ticker, s, samples, optionLegDict, current_spot_price), callback=simulator_price_collector_callback)

	pool.close()
	pool.join()
	mp_end_dt = datetime.datetime.now()

	# Parallelize this
	#for i in range(0, number_of_simulations):
		#print(str("Running simulation number: " + str(i + 1)))
		
		# TODO HERE PARALLELIZATION NEEDS TO TAKE PLACE

		#simulated_price_movement = s.get_sample_of_current_kde(samples)
		#simulated_price = np.empty([samples, 1])

		#simulated_price[0][0] = current_spot_price
		#for i in range(1, simulated_price.shape[0]):
		#	#if i == 0:
		#	#	simulated_price[i][0] = current_spot_price * (1 + simulated_price_movement[0][0])
		#	simulated_price[i][0] = simulated_price[i-1][0] * (1 + simulated_price_movement[i][0])

		#underlying_price_time_series_value = underlying_price_time_series_value.append(pd.DataFrame(simulated_price, columns=["simulated_price"])["simulated_price"])
		#price_dict = {}
		#price_dict[ticker] = simulated_price.reshape(samples)
		
		# TODO Get Price data stored somewhere to calculate the expected value as well

		# The number of simulations for the leg
		#strategy = Strategy(price_dict)

		#optionLegDict = {}
		#optionLegDict["ticker"] = ticker
		#optionLegDict["dividend_rate"] = dividend_rate
		#optionLegDict["option"] = option_type
		#optionLegDict["position_type"] = "Short"
		#optionLegDict["volatility"] = current_volatility
		#optionLegDict["strike_price"] = strike_price
		#optionLegDict["maturity_date"] = maturity_date
		#optionLegDict["risk_free_rate"] = risk_free_rate
		#optionLegDict["start_date"] = start_date
		
		#strategy.addLeg(optionLegDict)
		#strategy.simulate()
		#option_time_series_value = option_time_series_value.append(strategy.legs[0].simulated_price_data["value"])

	option_time_series_value.describe()
	break_even_threshold_price = option_time_series_value.iloc[0,0] # Price at time=0
	percentile_breakeven_at_end = stats.percentileofscore(option_time_series_value[underlying_price_time_series_value.shape[1]-1], break_even_threshold_price)
	
	expected_value_at_end = (option_time_series_value[underlying_price_time_series_value.shape[1]-1] * (1/number_of_simulations)).sum()
	expected_value_daily = (option_time_series_value * (1/number_of_simulations)).sum()
	percentile_breakeven_daily = stats.percentileofscore(expected_value_daily, break_even_threshold_price)
	
	underlying_price_time_series_value.describe()
	stock_price_estimate_lower_bound = underlying_price_time_series_value.quantile(0.05)
	stock_price_estimate_median = underlying_price_time_series_value.quantile(0.50)
	stock_price_estimate_upper_bound = underlying_price_time_series_value.quantile(0.95)

	option_price_estimate_lower_bound = option_time_series_value.quantile(0.05)
	option_price_estimate_expected_value = expected_value_daily
	option_price_estimate_median = option_time_series_value.quantile(0.50)
	option_price_estimate_upper_bound = option_time_series_value.quantile(0.95)
	option_price_estimate_upper_bound_75 = option_time_series_value.quantile(0.75)
	option_price_estimate_upper_bound_85 = option_time_series_value.quantile(0.85)

	stock_price_estimate_df = pd.DataFrame(data={"Lower Bound" : stock_price_estimate_lower_bound, "Upper Bound" : stock_price_estimate_upper_bound, "Median" : stock_price_estimate_median})

	plt.plot(stock_price_estimate_df['Lower Bound'], label='Lower Bound')
	plt.plot(stock_price_estimate_df['Median'], label='Median')
	plt.plot(stock_price_estimate_df['Upper Bound'], label='Upper Bound')
	plt.legend()
	plt.show()
		
	break_even_threshold_price_series = [break_even_threshold_price for i in range(0, date_difference)]
	option_price_estimate_df = pd.DataFrame(data={"Upper Bound 85%": option_price_estimate_upper_bound_85, "Upper Bound 75%": option_price_estimate_upper_bound_75, "Breakeven" : break_even_threshold_price_series, "Lower Bound 5%" : option_price_estimate_lower_bound, "Upper Bound 95%": option_price_estimate_upper_bound, "Median 50%" : option_price_estimate_median, "Expected" : option_price_estimate_expected_value})

	plt.plot(option_price_estimate_df['Lower Bound 5%'], label='Lower Bound 5%')
	plt.plot(option_price_estimate_df['Median 50%'], label='Median')
	plt.plot(option_price_estimate_df['Upper Bound 95%'], label='Upper Bound 95%')
	plt.plot(option_price_estimate_df['Expected'], label='Expected')
	plt.plot(option_price_estimate_df['Breakeven'], label='Breakeven')
	plt.plot(option_price_estimate_df['Upper Bound 75%'], label='Upper Bound 75%')
	plt.plot(option_price_estimate_df['Upper Bound 85%'], label='Upper Bound 85%')
	plt.legend()
	plt.show()

	# If expected simulated value is greater than the price at time 0, it is smarter to buy the call option vs. sell

