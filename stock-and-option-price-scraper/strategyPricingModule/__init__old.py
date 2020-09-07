import mibian
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import copy



GREEKS_DECIMALS = 8

class Strategy:
	def __init__(self):
		pass

	def advance_one_unit_of_time(self):
		pass

	def save_current_option_data_to_history(self):
		pass

class StrangleStrategy(Strategy):
	ESTIMATION_METHOD = "BS\\NO-DIVIDENDS"

	def __init__(self, call_strike_price, put_strike_price, current_underlying_price, days_to_expiration, current_underlying_volatility):
		self.current_price_of_underlying = current_underlying_price

		self.current_days_to_expiration = days_to_expiration 

		self.call_strike_price = call_strike_price
		self.put_strike_price = put_strike_price

		self.current_call = self.get_option_object(current_underlying_price, call_strike_price, days_to_expiration, current_underlying_volatility)
		self.current_put = self.get_option_object(current_underlying_price, put_strike_price, days_to_expiration, current_underlying_volatility)

		#self.initial_investment = current_call.callPrice + current_put.putPrice
		self.initial_investment = None

		self.call_history = pd.DataFrame(columns=["days_to_expiration", "call_price", "delta", "theta", "vega", "gamma", "rho"])
		self.put_history = pd.DataFrame(columns=["days_to_expiration", "put_price", "delta", "theta", "vega", "gamma", "rho"])
		self.profit_loss_history = pd.DataFrame(columns=["days_to_expiration", "initial_investment", "call_price", "put_price", "current_value", "profit"])

	def get_option_object(self, current_underlying_price, strike_price, days_to_expiration, current_underlying_volatility):
		# Please note, the underlying volatility is in the form of e.g. 0.50 while the function will want it as 50.
		return copy.deepcopy(mibian.BS([current_underlying_price, strike_price, 0.0, days_to_expiration], volatility=current_underlying_volatility * 100))
	
	def advance_one_unit_of_time(self, current_underlying_price, current_underlying_volatility):
		if self.current_days_to_expiration == 0:
			raise Exception(ValueError, "Trying to update option prices beyond expiration. Should not happen. Error in logic.")
		
		self.current_price_of_underlying = current_underlying_price
		self.current_underlying_volatility = current_underlying_volatility
		
		self.current_days_to_expiration = self.current_days_to_expiration - 1
		if self.current_days_to_expiration == 0:
			self.current_days_to_expiration = 0.01
		
		# Update call prices
		self.current_call = self.get_option_object(current_underlying_price, self.call_strike_price, self.current_days_to_expiration, current_underlying_volatility)
		self.current_put = self.get_option_object(current_underlying_price, self.put_strike_price, self.current_days_to_expiration, current_underlying_volatility)

		# Save the new values to the history
		self.save_current_option_data_to_history()

	def save_current_option_data_to_history(self):
		# This method needs to be called every time a new instance is created, 
		# the first line is not automatically saved.

		if self.initial_investment is None:
			self.initial_investment = self.current_call.callPrice + self.current_put.putPrice

		self.call_history = self.call_history.append(
			{"days_to_expiration" : self.current_days_to_expiration,
			"call_price": round(self.current_call.callPrice, 3), 
			"delta" : round(self.current_call.callDelta, GREEKS_DECIMALS), 
			"theta" : round(self.current_call.callTheta, GREEKS_DECIMALS), 
			"vega" : round(self.current_call.vega, GREEKS_DECIMALS), 
			"gamma" : round(self.current_call.gamma, GREEKS_DECIMALS),
			"rho" : round(self.current_call.callRho, GREEKS_DECIMALS),
		}, ignore_index=True)

		self.put_history = self.put_history.append( 
			{"days_to_expiration" : self.current_days_to_expiration,
			"put_price": round(self.current_put.putPrice, 3), 
			"delta" : round(self.current_put.putDelta, GREEKS_DECIMALS), 
			"theta" : round(self.current_put.putTheta, GREEKS_DECIMALS), 
			"vega" : round(self.current_put.vega, GREEKS_DECIMALS), 
			"gamma" : round(self.current_put.gamma, GREEKS_DECIMALS),
			"rho" : round(self.current_put.putRho, GREEKS_DECIMALS),
		}, ignore_index=True)

		self.profit_loss_history = self.profit_loss_history.append( 
			{"days_to_expiration" : self.current_days_to_expiration,
			"underlying_price" : self.current_price_of_underlying,
			"initial_investment" : round(self.initial_investment, 3),
			"call_price" : round(self.current_call.callPrice, 3),
			"put_price" : round(self.current_put.putPrice, 3),
			"current_value" : round(self.current_call.callPrice + self.current_put.putPrice, 3),
			"profit" : 100 * round(((self.current_call.callPrice + self.current_put.putPrice) - self.initial_investment)/self.initial_investment, 4)
		}, ignore_index=True)

class PriceBehaviorModel:
	#####################
	# Defines the behavior of the price of the asset 
	# at every time step
	#####################
	def __init__(self):
		self.behavior_type = "" # "Constant" or "Varies"
		self.behavior_varies_steps = {} # Key value pairs of ["timestep"] = ("Action", [params])
		self.behavior_constant_step = () # ("Action", [params])

	def initialize_step_change_pattern(self, step_change_time, step_change_to_asset, time_period, change_type = None):
		self.behavior_type = "Varies"
		for i in range(1, time_period+1):
			if i == step_change_time:
				#self.behavior_varies_steps[step_change_time] = ("constant_change_addition", [step_change_to_asset])
				self.behavior_varies_steps[step_change_time] = (change_type, [step_change_to_asset])
			else:
				self.behavior_varies_steps[i] = ("do_nothing", [])

	def initialize_normal_distribution_each_step_pattern(self, time_period, mean, std):
		self.behavior_type = "Constant"
		self.behavior_constant_step = ("normal_distribution", [mean, std])

	def get_next_price(self, current_time, current_price):
		if self.behavior_type == "Varies":
			if current_time in self.behavior_varies_steps:
				return self.evaluate_step(self.behavior_varies_steps[current_time], current_price)
			else:
				raise Exception(KeyError, "No pre-defined action for current step at: " + str(current_time))
		if self.behavior_type == "Constant":
			return self.evaluate_step(self.behavior_constant_step, current_price)
	
	def evaluate_step(self, action, current_price):
		# Calls the name of the action [0] with params [1]
		return getattr(self, "action_" + action[0])(action[1], current_price)

	def action_do_nothing(self, params, current_price):
		return current_price
	
	def action_constant_change_addition(self, params, current_price):
		# params = list of size one, the number to be added to current price
		return current_price + params[0]

	def action_constant_change_multiply(self, params, current_price):
		# params = list of size one, the number to be added to current price
		return current_price * params[0]

	def action_normal_distribution(self, params, current_price):
		# params is a list
		# first element: mean of the normal distribution
		# second element: standard deviation
		
		# it is assumed that the normal distribution returns the change for the unit time
		# hence the (1 + X) 

		return current_price * (1 + np.random.normal(params[0], params[1], 1))[0]

class Asset:
	#####################
	# Asset object holds all the information about the underlying
	# such as price, price, prive movement over unit time period, 
	# etc.
	#####################
	def __init__(self, asset_start_price, price_behavior, volatility, days_to_expiry = None):
		self.price_start = asset_start_price
		self.price_now = asset_start_price		
		self.price_behavior_model = price_behavior
		self.current_time = 1
		self.volatility_now = volatility
		self.days_to_expiry_current = days_to_expiry

		self.price_history = pd.DataFrame({'price' : [asset_start_price], 'days_to_expiry' : [days_to_expiry]})
		self.price_history.index.name = "time"
	
	def advance_one_unit_of_time(self):
		# Advances one unit time
		# updates prices based on the pricing model
		self.price_now = self.price_behavior_model.get_next_price(self.current_time, self.price_now)
		self.current_time = self.current_time + 1
		self.days_to_expiry_current = self.days_to_expiry_current - 1

		self.price_history = self.price_history.append({
										'price' : round(self.price_now, 2), 
										'days_to_expiry' : self.days_to_expiry_current},
										ignore_index=True)

class Simulation:
	##################
	# Simulation is the container for simulation itself
	# initializes all required assets and strategies
	##################
	def __init__(self):
		pass
	
	def initialize():
		pass

	def run(self):
		print("Starting the simulation.")
		for i in range(1, self.duration + 1):
			print("Currently at iteration: " + str(i))
			self.asset.advance_one_unit_of_time()
			self.strategy.advance_one_unit_of_time(self.asset.price_now, self.asset.volatility_now)
					
		print("Simulation finished.")

	def get_call_option_history(self):
		return self.strategy.call_history

	def get_put_option_history(self):
		return self.strategy.put_history

	def get_profit_loss_history(self):
		return self.strategy.profit_loss_history

	def get_asset_history(self):
		return self.asset.price_history

	def plot(self):
		asset_history = self.get_asset_history()
		call_option_history = self.get_call_option_history()
		put_option_history = self.get_put_option_history()
		profit_loss_history = self.get_profit_loss_history()		
		
		#f, axes = plt.subplots(3,2)
		f, axes = plt.subplots(nrows=7)

		# Asset price plot
		sns.lineplot(
					x = asset_history["days_to_expiry"], 
					y = asset_history["price"],
					marker="o",
					#ax=axes[0,0]
					ax=axes[0]
					)
		#axes[0,0].set_xticks(range(0, asset_history.shape[0]))
		axes[0].invert_xaxis()

		# Call price plot
		sns.lineplot(
					x = call_option_history["days_to_expiration"], 
					y = call_option_history["call_price"],
					marker="o",
					#ax=axes[1,0]
					ax=axes[1]
					)
		#axes[1,0].set_xticks(range(0, asset_history.shape[0]))
		axes[1].invert_xaxis()

		# Put price plot
		sns.lineplot(
					x = put_option_history["days_to_expiration"], 
					y = put_option_history["put_price"],
					marker="o",
					#ax=axes[2,0]
					ax=axes[2]
					)
		#axes[2,0].set_xticks(range(0, asset_history.shape[0]))
		axes[2].invert_xaxis()

		# Protif-loss chart 
		sns.lineplot(
					x = profit_loss_history["days_to_expiration"], 
					y = profit_loss_history["current_value"],
					marker="o",
					#ax=axes[2,0]
					ax=axes[3]
					)
		#axes[2,0].set_xticks(range(0, asset_history.shape[0]))
		axes[3].invert_xaxis()

		# Profit % chart 
		sns.lineplot(
					x = profit_loss_history["days_to_expiration"], 
					y = profit_loss_history["profit"],
					marker="o",
					#ax=axes[2,0]
					ax=axes[4]
					)
		#axes[2,0].set_xticks(range(0, asset_history.shape[0]))
		axes[4].invert_xaxis()

		# Greeks plot, call / put

		# Theta % chart 
		sns.lineplot(
					x = call_option_history["days_to_expiration"], 
					y = put_option_history["theta"],
					marker="o",
					#ax=axes[2,0]
					ax=axes[5]
					)
		#axes[2,0].set_xticks(range(0, asset_history.shape[0]))
		axes[5].invert_xaxis()

		# Theta % chart 
		sns.lineplot(
					x = call_option_history["days_to_expiration"], 
					y = call_option_history["theta"],
					marker="o",
					#ax=axes[2,0]
					ax=axes[6]
					)
		#axes[2,0].set_xticks(range(0, asset_history.shape[0]))
		axes[6].invert_xaxis()


		# Volatility plot 

		# Total P/L plot

		plt.show()

class StrangleSimulation(Simulation):
	def __init__(self, duration, asset_start_price, asset_volatility, put_strike_price, call_strike_price, asset_mean_returns_per_unit_time = None):
		self.duration = duration
		self.asset_start_price = asset_start_price
		self.put_strike_price = put_strike_price
		self.call_strike_price = call_strike_price
		self.asset_volatility = asset_volatility
		self.asset_mean_returns_per_unit_time = asset_mean_returns_per_unit_time

		self.asset = None
		self.price_behavior_model = None
		self.strategy = None

	def initialize(self, behaviorModel = None, addition = None, when = None):
		self.price_behavior_model = PriceBehaviorModel()
		if behaviorModel == "step_change":
			self.price_behavior_model.initialize_step_change_pattern(when, addition, self.duration, "constant_change_multiply")
		if behaviorModel == "Normal":
			self.price_behavior_model.initialize_normal_distribution_each_step_pattern(self.duration, self.asset_mean_returns_per_unit_time, self.asset_volatility)
		
		self.asset = Asset(self.asset_start_price, self.price_behavior_model, self.asset_volatility, self.duration)
		self.strategy = StrangleStrategy(self.call_strike_price, self.put_strike_price, self.asset.price_now, self.duration, self.asset_volatility)
		self.strategy.save_current_option_data_to_history()

def main():
	
	simulation1 = StrangleSimulation(21, 50, 25, 55, 55)
	simulation1.initialize(behaviorModel = "step_change", addition = 1.10, when = 14)
	simulation1.run()
	simulation1.plot()

	#asset_history = simulation1.get_asset_history()
	#call_option_history = simulation.get_call_option_history()
	#put_option_history = simulation.get_put_option_history()
	profit_loss_history = simulation1.get_profit_loss_history()		
	profit_loss_history

	profit_loss_history.to_csv("C:\\Users\\orent\\test100.csv")


	#price_behavior = PriceBehaviorModel()
	#price_behavior.initialize_normal_distribution_each_step_pattern(simulation_duration, 0.01, 0.1)

	#asset = Asset(asset_start_price, price_behavior)
	#asset.advance_one_unit_of_time()


	#####################
	# Plots start
	# plot plot plot

	# Subplot:
	# - Asset price
	# - Call price
	# - Put price
	# - Theta
	# - Volatility
	# - Total P/L



	#####################

main()