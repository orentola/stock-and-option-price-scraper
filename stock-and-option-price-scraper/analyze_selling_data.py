#run_name = run_name_generic + "_" + ticker + "_" + start_dt + "_" + str(tf) + "_" + str(sh) + "_" + str(ssd) + "_" + str(sl) + "_" + str(stte)
# file naming

file_naming = ["ticker", "start_dt", "threshold_timeframe", "threshold_change_abs", "spread_short_distance", "spread_length", "spread_time_to_expiry", "last_value"] # Note last value is not part of the filename

import matplotlib as plt
import pandas as pd
from scipy import stats
import datetime

from os import listdir
from os.path import isfile, join

path = "C:\\Users\\orent\\Documents\\SpreadStrategyResults\\"
run_name = "profit_loss_SPY_testing__"

files = [f for f in listdir(path) if isfile(join(path, f))]
files = [f for f in files if f.find(run_name) > -1]

profit_loss_paths = [f for f in files if f.find("profit_loss") > -1]
total_profit_paths = [f for f in files if f.find("total_profit") > -1]

total_profit_df = pd.DataFrame()

for p in profit_loss_paths:
	c = p.replace('.csv', '').replace(run_name, '').split("_")

	temp_df_profit_loss = pd.read_csv(join(path, p))
	temp_df_total_profit = pd.read_csv(join(path, p.replace("profit_loss", "total_profit_run")))

	c.append(temp_df_profit_loss["total_profit"].iloc[-1])

	c.append(temp_df_total_profit.shape[0])
	c.append(temp_df_total_profit.mean()["return"])
	c.append(round(temp_df_total_profit.quantile(0.05)["return"], 2))

	years_elapsed = ((datetime.datetime.now() - datetime.datetime.strptime(c[1], "%Y-%m-%d")).days / 365)
	c.append(temp_df_total_profit.shape[0] / ((datetime.datetime.now() - datetime.datetime.strptime(c[1], "%Y-%m-%d")).days / 365))
	
	total_return_per_tied_capital = ((temp_df_profit_loss["total_profit"].iloc[-1] * 100) / (float(c[5]) * 100))
	c.append(total_return_per_tied_capital)
	c.append(total_return_per_tied_capital/years_elapsed)

	#annualized_return = round((temp_df_total_profit.mean()["return"] / float(c[6])) * 252, 2)
	#c.append(annualized_return)

	new_row = pd.DataFrame([c], columns=file_naming + ["number_of_spreads", "spread_avg_return", "tail_risk_005", "number_of_spreads_per_year", "total_return_per_tied_capital", "yearly_return_per_tied_capital"])

	if total_profit_df.empty is True:
		total_profit_df = new_row
	else:
		total_profit_df = total_profit_df.append(new_row)
	total_profit_df.reset_index(inplace=True, drop=True)
total_profit_df.to_csv(join(path, "results", "total_profit_output_" + run_name + ".csv"))
