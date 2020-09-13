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

def getSimulatedOptionPriceForOneIteration(ticker, stockPriceService, samples, optionLegsDict, current_spot_price):
	# TODO: Add support for multi-leg options
	price_dict = {}	
	simulated_underlying_price_series = getSimulatedPriceForOneIteration(stockPriceService, samples, current_spot_price)
	price_dict[ticker] = simulated_underlying_price_series 

	strategy = Strategy(price_dict)
	for k, v in optionLegsDict.items():
		strategy.addLeg(v)
	
	strategy.simulate()
	
	#return (simulated_underlying_price_series, strategy.legs[0].simulated_price_data["value"])
	#return (simulated_underlying_price_series, strategy.get_profit_loss_and_values(), strategy.strategy_profit_loss.columns.values.tolist())
	# Return value is (underlying_price, (profit_loss, leg_values))
	return (simulated_underlying_price_series, strategy.get_profit_loss_and_values())

def getSimulatedPriceForOneIteration(stockPriceService, samples, current_spot_price):
		simulated_price_movement = stockPriceService.get_sample_of_current_kde(samples, False)
		simulated_price = np.empty([samples, 1])

		simulated_price[0][0] = current_spot_price
		for i in range(1, simulated_price.shape[0]):
			#if i == 0:
			#	simulated_price[i][0] = current_spot_price * (1 + simulated_price_movement[0][0])
			simulated_price[i][0] = simulated_price[i-1][0] * (1 + simulated_price_movement[i][0])

		return pd.DataFrame(simulated_price, columns=["simulated_price"])["simulated_price"]

def getOptionData(optionLeg, spot_price, advance_days, position_type, volatility=None):
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
	final_option["position_type"] = position_type
	final_option["time_to_expiry"] = maturity_date - calculation_date + 1 # +1 for taking the expiration day into account
	final_option["value"] = american_option.NPV()
	
	if position_type == "short":
		final_option["value"] = -1 * final_option["value"]

	# TODO, UPDATE GREEKS BASED ON POSITION TYPE (LONG/SHORT)
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
		self.strategy_profit_loss = pd.DataFrame()

	def simulate(self):
		for i in self.legs:
			i.simulate(self.underlying_price_data_sets[i.ticker])

	def save_current_option_data_to_history(self):
		# TODO
		pass
	
	def get_profit_loss_and_values(self):
		# Get max length of an option based on which simulate the thing
		
		profit_loss_dict = {}
		option_leg_value_dict = {}
		for l in self.legs:
			profit_loss_dict[l.name + "_profit_loss"] = l.simulated_price_data["profit_loss"]
			option_leg_value_dict[l.name + "_value"] = l.simulated_price_data["value"]
		
		self.strategy_profit_loss = pd.DataFrame(data=profit_loss_dict)
		self.strategy_profit_loss["total_profit_loss"] = self.strategy_profit_loss.apply(lambda row: self._get_profit_loss_sum_row(row), axis=1)

		# The first value of profit_loss should be the cost of entering the position

		return (self.strategy_profit_loss, option_leg_value_dict)
	
	def _get_profit_loss_sum_row(self, row):
		# row.values is a numpy array
		temp_value = 0.0
		for v in range(0,row.values.shape[0]):
			temp_value = temp_value + row.values[v]
		return temp_value

	def get_profit_loss_time_series(self):
		pass

	def addLeg(self, data):
		#self.legs.append(type("leg" + str(len(self.legs)), (OptionLeg, ), data))
		self.legs.append(OptionLeg(data))

class OptionLeg:
	def __init__(self, dict):
		for k, v in dict.items():
			setattr(self, k, v)

		self.simulated_price_data = pd.DataFrame(columns=["underlying_price", "position_type", "initial_value", "value", "profit_loss", "delta", "theta", "gamma", "vega", "time_to_expiry"])

	def advance_one_unit_of_time(self, underlying_price, underlying_volatility=None):
		self.latest_price_data_object = getOptionData(self, underlying_price, underlying_volatility)

	def simulate(self, simulated_ticker_price):
		print("Starting the simulation for " + self.ticker + " " + self.option)
		print("Reseting the simulated_price_data dataframe.")
		self.simulated_price_data.drop(self.simulated_price_data.index, inplace=True)
		temp_initial_value = 0.0

		# TODO: This should be optimized to not append data to df
		for i in range(0, simulated_ticker_price.shape[0]):
			# if return is None, we're done then TODO
			# i = 0 houses the price at first step

			temp_option_price_object = getOptionData(self, simulated_ticker_price[i], i, self.position_type, self.volatility)

			if i == 0:
				# Grab the initial price
				temp_initial_value = temp_option_price_object["value"]	

			temp_option_price_object["initial_value"] = temp_initial_value
			temp_option_price_object["profit_loss"] = (temp_option_price_object["value"] - temp_option_price_object["initial_value"])
			self.simulated_price_data = self.simulated_price_data.append(temp_option_price_object, ignore_index=True)

			#print("Current iteration: " + str(i) + ", ticker price: " + str(simulated_ticker_price[i]) + ", current option price: " + str(temp_option_price_object["value"]))

class StockPriceService:
	def __init__(self, data_path):
		with open(data_path , "r") as f:
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

	def get_sample_of_current_kde(self, size, set_self=True):
		if set_self is True:
			self.current_kde_sample = self.kde.sample(size)
			return self.current_kde_sample
		else:
			return self.kde.sample(size)

	def get_last_close_price(self, ticker):
		temp_stock = Stock.from_dict(self.data[ticker])
		return temp_stock.get_last_close_price()

