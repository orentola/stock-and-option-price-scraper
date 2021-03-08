import optionstrategypricingmodule
from datetime import datetime
import datetime
import numpy as np
import pandas as pd
import math

import QuantLib as ql 

STOCK_DATA_PATH = "C:\\Users\\orent\\Documents\\StockDataDownloader\\2021-03-07_20_08_one_time_run\\data.json"
VOLATILITY_LOOKBACK_PERIOD = 20

def get_option_price(optionLeg, current_date, spot_price, volatility=None):
	# volatility = None => leads to usage of the volatility from the optionLeg parameter
	
	# Params:
	# expiration_date = string, "YYYY-MM-DD"
	# spot_price = float, the price of the underlying
	# strike_price = float, the strike price of the option contract 
	# volatility = float, yearly volatility, either implied or realized/historical
	# dividend_rate = float, yearly dividends %
	# risk_free_rate = float, the risk free rate of money
	# start_date = string, "YYYY-MM-DD"

	calculation_date = ql.Date(datetime.datetime.strftime(current_date, '%Y-%m-%d'), '%Y-%m-%d')
	expiration_date = ql.Date(datetime.datetime.strftime(optionLeg.expiration_date, '%Y-%m-%d'), '%Y-%m-%d')
	
	if (calculation_date > expiration_date):
		print("Over expiration, skipping.")
		return None

	spot_price = spot_price
	strike_price = optionLeg.strike_price
	volatility = volatility if volatility is not None else optionLeg.volatility
	dividend_rate = optionLeg.dividend_rate

	if optionLeg.type == "Call":
		option_type = ql.Option.Call 
	elif optionLeg.type == "Put":
		option_type = ql.Option.Put
	else:
		raise "Option type not defined, exiting."
	
	risk_free_rate = optionLeg.risk_free_rate
	
	day_count = ql.Actual365Fixed()
	#day_count = ql.Business252()
	calendar = ql.UnitedStates()

	
	ql.Settings.instance().evaluationDate = calculation_date

	payoff = ql.PlainVanillaPayoff(option_type, strike_price)
	settlement = calculation_date

	am_exercise = ql.AmericanExercise(settlement, expiration_date)
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
	final_option["type"] = optionLeg.type
	final_option["position_type"] = optionLeg.short_long
	final_option["time_to_expiry"] = expiration_date - calculation_date + 1 # +1 for taking the expiration day into account
	final_option["value"] = american_option.NPV()

	if optionLeg.short_long == "short":
		final_option["value"] = -1 * final_option["value"]

	# TODO, UPDATE GREEKS BASED ON POSITION TYPE (LONG/SHORT)
	final_option["delta"] = american_option.delta()
	final_option["theta"] = american_option.theta()
	final_option["gamma"] = american_option.gamma()
	#final_option["vega"] = american_option.vega()

	return final_option

class OptionLeg():
	def __init__(self, strike_price, expiration_date, type, short_long, date_purchased, dividend_rate=None):
		self.strike_price = strike_price
		self.expiration_date = expiration_date
		self.type = type
		self.short_long = short_long
		self.profit = 0.0
		self.price_history = pd.DataFrame(index=pd.to_datetime([]), columns=['price', 'volatility', 'profit'])
		self.date_purchased = date_purchased
		self.price_at_start = None
		self.active = True
		self.dividend_rate = 0.00
		self.risk_free_rate = 0.015
		self.ended_ITM = None
		self.cash_requirement = None # Todo
	
	def update(self, date, price, volatility, add_skew=False):
		# add_skew can be used to adjust the volatility skew

		# calculate the current price
		temp_option = get_option_price(self, date, price, volatility)
		
		if self.price_at_start == None:
			self.price_at_start = temp_option["value"]
		
		self.price_history.loc[date] = {'price' : temp_option["value"], 'volatility' : volatility, 'profit' : temp_option["value"] - self.price_at_start}

		if date == self.expiration_date:
			self.ended_ITM = True if temp_option["value"] > 0.0 else False
			self.active = False
	
	def to_json(self):
		output_dict = {}
		output_dict["strike_price"] = self.strike_price
		output_dict["expiration_date"] = self.expiration_date
		output_dict["type"] = self.type
		output_dict["short_long"] = self.short_long
		output_dict["profit"] = self.profit
		output_dict["price_history"] = self.price_history.to_json()
		# TODO



def main():
	# Spread details

	spread_short_distance = 0.10
	spread_length = 5
	spread_time_to_expiry = 45
	
	start_dt = '2000-05-26'
	#end_dt = '2000-08-26'
	end_dt = datetime.now().strftime("%Y-%m-%d")
	threshold_timeframes = [10]
	threshold_shift_abs = [0.03]

	for tf in threshold_timeframes:
		for s in threshold_shift_abs:

			s = optionstrategypricingmodule.StockPriceService(STOCK_DATA_PATH)
			price_history = s.get_price_history("IWM")

			price_history = price_history[start_dt : end_dt]

			price_history['threshold_price'] = price_history['Close'].shift(threshold_timeframe)
			price_history['threshold_percentage_change'] = price_history.apply(lambda row : (row.Close - row.threshold_price) / row.threshold_price, axis=1)
			price_history['action'] = price_history.apply(lambda row: "call" if row.threshold_percentage_change > threshold_shift_abs else ("put" if row.threshold_percentage_change < -1 * threshold_shift_abs else np.nan), axis=1 )

			price_history['previous_close'] = price_history['Close'].shift(1)
			price_history['change_per_day'] = price_history.apply(lambda row: (row.Close - row.previous_close) / row.previous_close, axis = 1 )

			#price_history.to_csv("C:\\Users\\orent\\Documents\\testdata.csv")

			option_legs = []
			volatility_counter = 0

			for index, row in price_history.iterrows():	
				if volatility_counter < VOLATILITY_LOOKBACK_PERIOD-1:
					volatility_counter = volatility_counter + 1
					continue

				if row.action == np.nan:
					continue

				current_date = index
				if row.action == 'call':
					option_legs.append(OptionLeg(row.Close * (1 + spread_short_distance), current_date + datetime.timedelta(days=spread_time_to_expiry), 'Call', 'short', current_date ))
					option_legs.append(OptionLeg(row.Close * (1 + spread_short_distance) + spread_length, current_date + datetime.timedelta(days=spread_time_to_expiry), 'Call', 'long', current_date ))
				elif row.action == 'put':
					option_legs.append(OptionLeg(row.Close * (1 - spread_short_distance), current_date + datetime.timedelta(days=spread_time_to_expiry), 'Put', 'short', current_date ))
					option_legs.append(OptionLeg(row.Close * (1 - spread_short_distance) - spread_length, current_date + datetime.timedelta(days=spread_time_to_expiry), 'Put', 'long', current_date ))
	
			# TODO:
			# Validate the option leg list so it goes correctly
	
			#rolling_volatility = pd.DataFrame(columns=["price"], index=pd.to_datetime([]))
			rolling_volatility = pd.DataFrame(columns=["price_change"])

			for index, row in price_history.iterrows():
				rolling_volatility = rolling_volatility.append({"price_change" : row.change_per_day}, ignore_index=True)

				if rolling_volatility.shape[0] < VOLATILITY_LOOKBACK_PERIOD:
					# to ensure we can calculate volatility correctly
					continue

				for leg in option_legs:
					if index < leg.date_purchased:
						# the leg has not been purchased yet
						continue
					if index > leg.expiration_date:
						# if we are past the expiration date
						continue
					current_volatility = rolling_volatility.std()[0] * math.sqrt(252)
					leg.update(index, row.Close, current_volatility, False)
		
				rolling_volatility.drop(0, inplace=True)
				rolling_volatility.reset_index(drop=True, inplace=True)

			# calculate profit loss per day
			profit_loss = pd.DataFrame(index=price_history.index)
			profit_loss['profit'] = [0.0 for x in range(0, profit_loss.index.shape[0])]

			for o in option_legs:
				for index, row in o.price_history.iterrows():
					profit_loss.loc[index] = { 'profit' : profit_loss.loc[index].profit + row.profit }