import mibian
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

GREEKS_DECIMALS = 8
STOCK_DATA_PATH = "C:\\Users\\orent\\Documents\\StockDataDownloader\\2020-09-07_20_18_one_time_run\\data.json"


def getOptionData(optionLeg, spot_price, advance_days, volatility=None):
	# volatility = None => leads to usage of the volatility from the optionLeg parameter
	
	# Params:
	# maturity_date = string, "YYYY-MM-DD"
	# spot_price = float, the price of the underlying
	# strike_price = float, the strike price of the option contract 
	# volatility = float, yearly volatility, either implied or realized/historical
	# dividend_rate = float, yearly dividends %
	# risk_free_rate = float, the risk free rate of money
	# start_date = string, "YYYY-MM-DD"
	
	current_date = datetime.datetime.strptime(optionLeg.start_date, '%Y-%m-%d') + datetime.timedelta(days=advance_days)

	calculation_date = ql.Date(datetime.datetime.strftime(current_date, '%Y-%m-%d'), '%Y-%m-%d')
	maturity_date = ql.Date(optionLeg.maturity_date, '%Y-%m-%d')
	
	if (calculation_date == maturity_date):
		print("At expiration, skipping.")
		return None

	spot_price = spot_price
	strike_price = optionLeg.strike_price
	volatility = volatility if volatility is not None else optionLeg.volatility
	dividend_rate = optionLeg.dividend_rate

	if optionLeg.option == "Call":
		option_type = ql.Option.Call 
	elif optionLeg.option == "Put":
		option_type = ql.Option.Put
	else:
		raise "Option type not defined, exiting."
	
	risk_free_rate = optionLeg.risk_free_rate
	
	day_count = ql.Actual365Fixed()
	calendar = ql.UnitedStates()

	
	ql.Settings.instance().evaluationDate = calculation_date

	payoff = ql.PlainVanillaPayoff(option_type, strike_price)
	settlement = calculation_date

	am_exercise = ql.AmericanExercise(settlement, maturity_date)
	american_option = ql.VanillaOption(payoff, am_exercise)

	#eu_exercise = ql.EuropeanExercise(maturity_date)
	#european_option = ql.VanillaOption(payoff, eu_exercise)

	spot_handle = ql.QuoteHandle(
		ql.SimpleQuote(spot_price)
	)
	flat_ts = ql.YieldTermStructureHandle(
		ql.FlatForward(calculation_date, risk_free_rate, day_count)
	)
	dividend_yield = ql.YieldTermStructureHandle(
		ql.FlatForward(calculation_date, dividend_rate, day_count)
	)
	flat_vol_ts = ql.BlackVolTermStructureHandle(
		ql.BlackConstantVol(calculation_date, calendar, volatility, day_count)
	)
	bsm_process = ql.BlackScholesMertonProcess(spot_handle, 
											   dividend_yield, 
											   flat_ts, 
											   flat_vol_ts)

	steps = 200
	binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", steps)
	american_option.setPricingEngine(binomial_engine)
	
	final_option = {}
	final_option["underlying_price"] = spot_price
	final_option["type"] = optionLeg.option
	final_option["time_to_expiry"] = maturity_date - calculation_date + 1 # +1 for taking the expiration day into account
	final_option["value"] = american_option.NPV()
	final_option["delta"] = american_option.delta()
	final_option["theta"] = american_option.theta()
	final_option["gamma"] = american_option.gamma()
	#final_option["vega"] = american_option.vega()

	return final_option

class Strategy:
	# The purpose of this class is to hold all the legs of the strategy
	# Then there is another class that will handle the simulation

	def __init__(self, price_dict):
		self.legs = [] # Option legs in this strategy
		self.underlying_price_data_sets = price_dict # Dictionary of pandas dataframes that has 
		self.underlying_volatility_data_sets = {}

	def simulate(self):
		# Parallelize this TODO
		for i in self.legs:
			i.simulate(self.underlying_price_data_sets[i.ticker])

	def save_current_option_data_to_history(self):
		# TODO
		pass
	
	def calculate_profit_loss_over_time(self):
		# Get max length of an option based on which simulate the thing

		for i in range(0, self.max_length):
			self.advance_one_unit_of_time()
	
	def get_profit_loss_time_series(self):
		pass

	def addLeg(self, data):
		#self.legs.append(type("leg" + str(len(self.legs)), (OptionLeg, ), data))
		self.legs.append(OptionLeg(data))

class OptionLeg:
	def __init__(self, dict):
		for k, v in dict.items():
			setattr(self, k, v)

		#self.ticker = ""
		#self.dividend_rate = ""
		#self.option = ""
		#self.option_type = "" # short or long
		#self.volatility = "" 
		#self.strike_price = ""
		#self.spot_price = ""
		#self.maturity_date = ""
		#self.start_date = ""
		#self.risk_free_rate = ""
		#self.latest_price_data = ""
		#self.price_data_history = ""
		#self.underlying_price_data = ""
		#self.underlying_volatility_data = ""
		#self.latest_start_date = ""
		self.simulated_price_data = pd.DataFrame(columns=["underlying_price", "value", "delta", "theta", "gamma", "vega", "time_to_expiry"])

	def advance_one_unit_of_time(self, underlying_price, underlying_volatility=None):
		self.latest_price_data_object = getOptionData(self, underlying_price, underlying_volatility)

	def simulate(self, simulated_ticker_price):
		print("Starting the simulation for " + self.ticker + " " + self.option)
		print("Reseting the simulated_price_data dataframe.")
		self.simulated_price_data.drop(self.simulated_price_data.index, inplace=True)

		# Get next business day 

		for i in range(0, simulated_ticker_price.shape[0]):
			# if return is None, we're done then TODO
			temp_option_price_object = getOptionData(self, simulated_ticker_price[i], i, self.volatility)
			self.simulated_price_data = self.simulated_price_data.append(temp_option_price_object, ignore_index=True)

			#print("Current iteration: " + str(i) + ", ticker price: " + str(simulated_ticker_price[i]) + ", current option price: " + str(temp_option_price_object["value"]))

class StockPriceService:
	def __init__(self):
		with open(STOCK_DATA_PATH , "r") as f:
			self.data = json.load(f)

	def plot_histogram(self, ticker, include_current_kde_sample=False, include_normal_dist=False):
		temp_stock = Stock.from_dict(self.data[ticker])
		temp_volatility = Volatility(temp_stock.stock_price_history, "Close")

		plt.hist(temp_volatility.time_series_daily['change'], bins=200, density=True, label="data")
		if include_current_kde_sample == True:
			plt.hist(self.current_kde_sample, bins=200, density=True, label="KDE", alpha=0.2)
		plt.legend(loc='upper left')
		plt.show()

		if include_normal_dist == True:
			plt.hist(temp_volatility.time_series_daily['change'], bins=200, density=True, label="data")
			plt.hist(self.get_sample_of_current_normal_distribution(10000), bins=200, density=True, label="Normal Estimate", alpha=0.2)
			plt.legend(loc='upper left')
			plt.show()


	def get_volatility(self, ticker, periods):
		temp_stock = Stock.from_dict(self.data[ticker])
		return Volatility(temp_stock.stock_price_history, "Close").get_daily(periods)

	def get_daily_change(self, ticker, periods=None):
		temp_stock = Stock.from_dict(self.data[ticker])
		return Volatility(temp_stock.stock_price_history, "Close").time_series_daily['change']

	def fit_normal_dist(self, ticker):
		self.mu, self.std = norm.fit(self.get_daily_change(ticker))

	def get_sample_of_current_normal_distribution(self, samples):
		return np.random.normal(self.mu, self.std, samples)

	def make_kde(self, ticker, bandwidth=0.002):
		daily_change_data = self.get_daily_change(ticker).to_numpy().reshape(-1,1)
		if bandwidth == None:
			params = {'bandwidth': np.linspace(-0.1, 0.1, 50)}
			grid = GridSearchCV(KernelDensity(), params)
			grid.fit(daily_change_data)
			print(str(grid.best_estimator_.bandwidth))
			self.kde = grid.best_estimator_
		else:
			self.kde = KernelDensity(bandwidth=bandwidth)
			self.kde.fit(daily_change_data)
		self.current_kde_ticker = ticker
		return "Success"

	def get_sample_of_current_kde(self, size):
		self.current_kde_sample = self.kde.sample(size)
		return self.current_kde_sample

	def get_last_close_price(self, ticker):
		temp_stock = Stock.from_dict(self.data[ticker])
		return temp_stock.get_last_close_price()



def main():
	ticker = "WORK"
	strike_price = 20
	number_of_simulations = 100
	dividend_rate = 0.00
	risk_free_rate = 0.008

	start_date = "2020-05-01" # YYYY-MM-DD
	maturity_date = "2020-06-10" # YYYY-MM-DD

	date_difference = (datetime.datetime.strptime(maturity_date, "%Y-%m-%d") - datetime.datetime.strptime(start_date, "%Y-%m-%d")).days
	# Sample for the duration
	samples = date_difference

	s = StockPriceService()
	s.make_kde(ticker)
	s.get_sample_of_current_kde(10000)
	s.fit_normal_dist(ticker)
	s.plot_histogram(ticker, True, True)

	current_spot_price = s.get_last_close_price(ticker)
	current_volatility_daily = s.get_volatility(ticker, 30) / 100
	current_volatility = current_volatility_daily * math.sqrt(365) # THIS MAY NEED TO BE 252
	#current_volatility = 0.93
	
	option_time_series_value = pd.DataFrame()
	underlying_price_time_series_value = pd.DataFrame()

	for i in range(0, number_of_simulations):
		print(str("Running simulation number: " + str(i + 1)))

		simulated_price_movement = s.get_sample_of_current_kde(samples)
		simulated_price = np.empty([samples, 1])

		simulated_price[0][0] = current_spot_price
		for i in range(1, simulated_price.shape[0]):
			#if i == 0:
			#	simulated_price[i][0] = current_spot_price * (1 + simulated_price_movement[0][0])
			simulated_price[i][0] = simulated_price[i-1][0] * (1 + simulated_price_movement[i][0])

		underlying_price_time_series_value = underlying_price_time_series_value.append(pd.DataFrame(simulated_price, columns=["simulated_price"])["simulated_price"])
		price_dict = {}
		price_dict[ticker] = simulated_price.reshape(samples)
		
		# TODO Get Price data stored somewhere to calculate the expected value as well

		# The number of simulations for the leg
		strategy = Strategy(price_dict)

		optionLegDict = {}
		optionLegDict["ticker"] = ticker
		optionLegDict["dividend_rate"] = dividend_rate
		optionLegDict["option"] = "Put"
		optionLegDict["position_type"] = "Short"
		optionLegDict["volatility"] = current_volatility
		optionLegDict["strike_price"] = strike_price
		optionLegDict["maturity_date"] = maturity_date
		optionLegDict["risk_free_rate"] = risk_free_rate
		optionLegDict["start_date"] = start_date
		
		strategy.addLeg(optionLegDict)
		strategy.simulate()
		option_time_series_value = option_time_series_value.append(strategy.legs[0].simulated_price_data["value"])

	option_time_series_value.describe()
	break_even_threshold_price = option_time_series_value.iloc[0,0] # Price at time=0
	percentile_breakeven_at_end = stats.percentileofscore(option_time_series_value[simulated_price.shape[0]-1], break_even_threshold_price)
	
	expected_value_at_end = (option_time_series_value[simulated_price.shape[0]-1] * (1/number_of_simulations)).sum()
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

	option_price_estimate_df = pd.DataFrame(data={"Lower Bound" : option_price_estimate_lower_bound, "Upper Bound": option_price_estimate_upper_bound, "Median" : option_price_estimate_median, "Expected" : option_price_estimate_expected_value})

	stock_price_estimate_df = pd.DataFrame(data={"Lower Bound" : stock_price_estimate_lower_bound, "Upper Bound" : stock_price_estimate_upper_bound, "Median" : stock_price_estimate_median})

	plt.plot(stock_price_estimate_df['Lower Bound'], label='Lower Bound')
	plt.plot(stock_price_estimate_df['Median'], label='Median')
	plt.plot(stock_price_estimate_df['Upper Bound'], label='Upper Bound')
	plt.legend()
	plt.show()
	
	plt.plot(option_price_estimate_df['Lower Bound'], label='Lower Bound')
	plt.plot(option_price_estimate_df['Median'], label='Median')
	plt.plot(option_price_estimate_df['Upper Bound'], label='Upper Bound')
	plt.plot(option_price_estimate_df['Expected'], label='Expected')
	plt.legend()
	plt.show()

	# If expected simulated value is greater than the price at time 0, it is smarter to buy the call option vs. sell
