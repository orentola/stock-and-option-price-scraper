import optionstrategypricingmodule
from datetime import datetime
import datetime
import numpy as np
import pandas as pd
import math
import multiprocessing as mp

import QuantLib as ql 

STOCK_DATA_PATH = "C:\\Users\\orent\\Documents\\StockDataDownloader\\2021-03-10_20_40_one_time_run\\data.json"
VOLATILITY_LOOKBACK_PERIOD = 20
OUTPUT_PATH = "C:\\Users\\orent\\Documents\\\SpreadStrategyResults\\"
ENABLE_MULTIPROCESSING = False

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
		self.price_history = pd.DataFrame(index=pd.to_datetime([]), columns=['price', 'volatility', 'value'])
		self.date_purchased = date_purchased
		self.price_at_start = None
		self.active = True
		self.dividend_rate = 0.00
		self.risk_free_rate = 0.015
		self.ended_ITM = None
		self.cash_requirement = None # Todo
		self.total_profit = 0.0
	
	def update(self, date, price, volatility, add_skew=False):
		# calculate the current price
		
		if (self.expiration_date == date):
			#self.ended_ITM = True if self.price_history.iloc[-1]["value"] > 0.0 else False
			self.active = False
			return None

		temp_option = get_option_price(self, date, price, volatility)

		if self.price_at_start == None:
			self.price_at_start = temp_option["value"]
		
		self.price_history.loc[date] = {'price' : temp_option["value"], 'volatility' : volatility, 'value' : temp_option["value"] - self.price_at_start}
		self.total_profit = temp_option["value"] - self.price_at_start

		# if calculated at expiration, the function to calculate the value provides a 0.0. hence we stop at the day before expiration. good enough approximation.

			
	def to_json(self):
		output_dict = {}
		output_dict["strike_price"] = self.strike_price
		output_dict["expiration_date"] = self.expiration_date
		output_dict["type"] = self.type
		output_dict["short_long"] = self.short_long
		output_dict["profit"] = self.profit
		output_dict["price_history"] = self.price_history.to_json()
		# TODO

def sum_values_from_date_onwards(options, df_index):
	profit_loss = pd.DataFrame(index=df_index)
	profit_loss["profit_only"] = [0.0 for x in range(0, df_index.shape[0])]
	profit_loss['value_only'] = [0.0 for x in range(0, df_index.shape[0])]

	for o in options:
		#for i1, row in o.price_history[:-1].iterrows():
		for i1, row in o.price_history.iterrows():
			profit_loss.loc[i1].value_only = profit_loss.loc[i1].value_only + row.value
				
		# If expiration is in the future that is not part of the profit loss dataframe, let's skip and continue
		if o.expiration_date >= profit_loss.index[profit_loss.index.shape[0]-1]:
			continue

		for i, row in profit_loss[o.expiration_date:].iterrows():
			try:
				profit_loss.loc[i].profit_only = profit_loss.loc[i].profit_only + o.total_profit
			except:
				print("Exception caught.")
	return profit_loss

profit_loss_result = pd.DataFrame()
def sum_values_from_data_onwards_callback(result):
	global profit_loss_result
	if profit_loss_result.empty == True:
		profit_loss_result = result
	else:
		profit_loss_result["profit_only"] = profit_loss_result["profit_only"] + result["profit_only"]
		profit_loss_result["value_only"] = profit_loss_result["value_only"] + result["value_only"]

def split_list(list, n):
	n = max(1, n)
	return (list[i:i+n] for i in range(0, len(list), n))

def main():
	# Spread details
	print("Starting the run at: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
	
	#spread_short_distances = [0.10, 0.15] # How many percentage points from the last value (e.g. 0.15 means 15% below for short put and 15% above call)
	#spread_lengths = [5, 10, 20, 30]
	#spread_time_to_expirys = [30, 60, 90, 120]
	
	spread_short_distances = [0.10] # How many percentage points from the last value (e.g. 0.15 means 15% below for short put and 15% above call)
	spread_lengths = [10]
	spread_time_to_expirys = [120]

	ticker = "SPY"
	#run_name_generic = "SPY_testing_"
	run_name_generic = "Debug_run_"
	
	start_dt = '2000-01-01'
	#end_dt = '2000-08-26'
	end_dt = datetime.datetime.now().strftime("%Y-%m-%d")
	
	#threshold_timeframes = [5, 10, 20] # Lookback period for triggering decision
	#threshold_shift_abs = [0.02, 0.04, 0.06, 0.08] # How much the price needs to change in the threshold_timeframes timeframe

	threshold_timeframes = [5] # Lookback period for triggering decision
	threshold_shift_abs = [0.08] # How much the price needs to change in the threshold_timeframes timeframe

	for sl in spread_lengths:
		for stte in spread_time_to_expirys:
			for ssd in spread_short_distances:
				for tf in threshold_timeframes:
					for sh in threshold_shift_abs:

						run_name = run_name_generic + "_" + ticker + "_" + start_dt + "_" + str(tf) + "_" + str(sh) + "_" + str(ssd) + "_" + str(sl) + "_" + str(stte)
						print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " - " + "Currently running: " + run_name)

						s = optionstrategypricingmodule.StockPriceService(STOCK_DATA_PATH)
						price_history = s.get_price_history(ticker)

						price_history = price_history[start_dt : end_dt]

						price_history['threshold_price'] = price_history['Close'].shift(tf)
						price_history['threshold_percentage_change'] = price_history.apply(lambda row : (row.Close - row.threshold_price) / row.threshold_price, axis=1)
						price_history['action'] = price_history.apply(lambda row: "call" if row.threshold_percentage_change > sh else ("put" if row.threshold_percentage_change < -1 * sh else np.nan), axis=1 )

						price_history['previous_close'] = price_history['Close'].shift(1)
						price_history['change_per_day'] = price_history.apply(lambda row: (row.Close - row.previous_close) / row.previous_close, axis = 1 )
						price_history['volatility'] = price_history['change_per_day'].rolling(VOLATILITY_LOOKBACK_PERIOD).std() * math.sqrt(252)
						#price_history.to_csv("C:\\Users\\orent\\Documents\\testdata.csv")
						cash_requirements = pd.DataFrame(columns=['cash'], index=price_history.index)
						cash_requirements['cash'] = [0.0 for x in range(0,cash_requirements.shape[0])]
					
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
								option_legs.append(OptionLeg(row.Close * (1 + ssd), current_date + datetime.timedelta(days=stte-1), 'Call', 'short', current_date ))
								option_legs.append(OptionLeg(row.Close * (1 + ssd) + sl, current_date + datetime.timedelta(days=stte-1), 'Call', 'long', current_date ))
								for i in range(0, stte):
									try:
										cash_requirements.loc[current_date + datetime.timedelta(days=i)] = cash_requirements.loc[current_date + datetime.timedelta(days=i)] + sl * 100
									except KeyError:
										continue
										#print("This is probably a weekend, no panic")
							elif row.action == 'put':
								option_legs.append(OptionLeg(row.Close * (1 - ssd), current_date + datetime.timedelta(days=stte-1), 'Put', 'short', current_date ))
								option_legs.append(OptionLeg(row.Close * (1 - ssd) - sl, current_date + datetime.timedelta(days=stte-1), 'Put', 'long', current_date ))
								for i in range(0, stte):
									try:
										cash_requirements.loc[current_date + datetime.timedelta(days=i)] = cash_requirements.loc[current_date + datetime.timedelta(days=i)] + sl * 100
									except KeyError:
										continue
										#print("This is probably a weekend, no panic")			
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
						profit_loss['profit_only'] = [0.0 for x in range(0, profit_loss.index.shape[0])]
						profit_loss['value_only'] = [0.0 for x in range(0, profit_loss.index.shape[0])]

						if ENABLE_MULTIPROCESSING:
							mp_start_dt = datetime.datetime.now()
							num_of_cores = mp.cpu_count()-1
							pool = mp.Pool(num_of_cores)
							# Do all the tasks, join, and then sum everything up and order by index
							option_sets = split_list(option_legs, num_of_cores)
							#for i in range(0, min(num_of_cores, len(option_sets)):
							for s in option_sets:
								pool.apply_async(sum_values_from_date_onwards, args=(s, profit_loss.index), callback=sum_values_from_data_onwards_callback)
							print("Trying to close the pool..")
							pool.close()
							pool.join()
							mp_end_dt = datetime.datetime.now()
							profit_loss = profit_loss_result

						else:
							for o in option_legs:
								#for i1, row in o.price_history[:-1].iterrows():
								for i1, row in o.price_history.iterrows():
									profit_loss.loc[i1].value_only = profit_loss.loc[i1].value_only + row.value
				
								# If expiration is in the future that is not part of the profit loss dataframe, let's skip and continue
								if o.expiration_date >= profit_loss.index[profit_loss.index.shape[0]-1]:
									continue

								# Ensure the profit is added cumulatively from the expiration onwards
								for i2, row in profit_loss[o.expiration_date:].iterrows():
									try:
										profit_loss.loc[i2].profit_only = profit_loss.loc[i2].profit_only + o.total_profit
									except:
										print("Exception caught.")

						profit_loss["total_profit"] = profit_loss["profit_only"] + profit_loss["value_only"]
						profit_loss["underlying_price"] = price_history["Close"]
					
						profit_loss.to_csv(OUTPUT_PATH + "profit_loss_" + run_name + ".csv")
			
						total_profit_data = []
						spread_identifier = 0
						for o in option_legs:
							if spread_identifier == 0:
								temp_spread = o
								spread_identifier = spread_identifier + 1
								continue

							profit_spread = o.total_profit + temp_spread.total_profit
							total_profit_data.append(profit_spread)
							spread_identifier = 0
						total_profit_data_df = pd.DataFrame(total_profit_data, columns=["profit"])
						total_profit_data_df["return"] = (total_profit_data_df["profit"] / sl) * 100
						total_profit_data_df.to_csv(OUTPUT_PATH + "total_profit_run_" + run_name + ".csv")

if __name__ == '__main__':
	main()