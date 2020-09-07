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

from scipy.stats import shapiro
from scipy.stats import norm

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from matplotlib import pyplot
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot

import numpy as np

GREEKS_DECIMALS = 8
STOCK_DATA_PATH = "C:\\Users\\orent\\Documents\\StockDataDownloader\\2020-08-12_23_07_one_time_run\\data.json"


def getOptionData(optionLeg, spot_price, volatility=None):
	# volatility = None => leads to usage of the volatility from the optionLeg parameter
	
	# Params:
	# maturity_date = string, "YYYY-MM-DD"
	# spot_price = float, the price of the underlying
	# strike_price = float, the strike price of the option contract 
	# volatility = float, yearly volatility, either implied or realized/historical
	# dividend_rate = float, yearly dividends %
	# risk_free_rate = float, the risk free rate of money
	# start_date = string, "YYYY-MM-DD"
		
	maturity_date = ql.Date(optionLeg.maturity_date, '%Y-%m-%d')
	spot_price = spot_price
	strike_price = optionLeg.strike_price
	volatility = volatility if volatility is not None else optionLeg.volatility
	dividend_rate = optionLeg.dividend_rate
	start_date = optionLeg.start_date

	if optionLeg.option == "Call":
		option_type = ql.Option.Call 
	elif optionLeg.option == "Put":
		option_type = ql.Option.Put
	else:
		raise "Option type not defined, exiting."
	
	risk_free_rate = optionLeg.risk_free_rate
	
	day_count = ql.Actual365Fixed()
	calendar = ql.UnitedStates()

	calculation_date = ql.Date(start_date, '%Y-%m-%d')
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
	final_option["type"] = option
	final_option["value"] = american_option.NPV()
	final_option["delta"] = american_option.delta()
	final_option["theta"] = american_option.theta()
	final_option["gamma"] = american_option.gamma()
	final_option["vega"] = american_option.vega()

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
			i.simulate()

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


class OptionLeg:
	def __init__(self):

		self.ticker = ""
		self.dividend_rate = ""
		self.option = ""
		self.option_type = "" # short or long
		self.volatility = "" 
		self.strike_price = ""
		self.spot_price = ""
		self.maturity_date = ""
		#self.start_date = ""
		self.risk_free_rate = ""
		self.latest_price_data = ""
		self.price_data_history = ""
		self.underlying_price_data = ""
		self.underlying_volatility_data = ""
		self.latest_start_date = ""

	def advance_one_unit_of_time(self, underlying_price, underlying_volatility=None):
		self.latest_price_data_object = getOptionData(self, underlying_price, underlying_volatility)

	def simulate(self, infer_volatility ):

		# Get next business day 

		for i in self.underlying_price_data:


class StockPriceService:
	def __init__(self):
		with open(STOCK_DATA_PATH , "r") as f:
			self.data = json.load(f)

	def plot_histogram(self, ticker, include_current_kde_sample=False):
		temp_stock = Stock.from_dict(self.data[ticker])
		temp_volatility = Volatility(temp_stock.stock_price_history, "Close")

		plt.hist(temp_volatility.time_series_daily['change'], bins=250, density=True, label="data")
		if include_current_kde_sample == True:
			plt.hist(self.current_kde_sample, bins=250, density=True, label="KDE", alpha=0.2)
		plt.legend(loc='upper left')
		plt.show()

	def get_daily_change(self, ticker):
		temp_stock = Stock.from_dict(self.data[ticker])
		return Volatility(temp_stock.stock_price_history, "Close").time_series_daily['change']

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




def main():
	samples = 300

	s = StockPriceService()
	s.make_kde("MSFT")

	current_spot_price = 214.25
	simulated_price_movement = s.get_sample_of_current_kde(samples)
	simulated_price = np.empty([samples, 1])

	for i in range(0, simulated_price.shape[0]):
		if i == 0:
			simulated_price[i][0] = current_spot_price * (1 + simulated_price_movement[0][0])
		else:
			simulated_price[i][0] = simulated_price[i-1][0] * (1 + simulated_price_movement[i][0])
	price_dict = {}
	price_dict["MSFT"] = simulated_price.reshape(samples)
	
	# The number of simulations for the leg
	strategy = Strategy(price_dict)

	strategy.addLeg()
	strategy.addLeg()
	strategy.addLeg()

	# Generate the underlying datasets
	# The underlying datasets will be dictionaries containing the dataframes
	# i.e. data["WIX"] = pd.DataFrame(price_data)

	# Assign the datasets to Strategy object

	strategy.simulate()	


