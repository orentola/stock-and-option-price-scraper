#run_name = run_name_generic + "_" + ticker + "_" + start_dt + "_" + str(tf) + "_" + str(sh) + "_" + str(ssd) + "_" + str(sl) + "_" + str(stte)
# file naming

file_naming = ["ticker", "start_dt", "threshold_timeframe", "threshold_change_abs", "spread_short_distance", "spread_length", "spread_time_to_expiry", "last_value"] # Note last value is not part of the filename

import matplotlib as plt
import pandas as pd

from os import listdir
from os.path import isfile, join

path = "C:\\Users\\orent\\Documents\\SpreadStrategyResults\\"
run_name = "profit_loss_SPY_testing__"

files = [f for f in listdir(path) if isfile(join(path, f))]
files = [f for f in files if f.find(run_name) > -1]

profit_loss_paths = [f for f in files if f.find("profit_loss") > -1]
total_profit_paths = [f for f in files if f.find("total_profit") > -1]

for p in profit_loss_paths:
	c = p.replace('.csv', '').replace(run_name, '').split("_")

	temp_df = pd.read_csv(join(path, p))
	c.append(temp_df["total_profit"].iloc[-1])

	new_row = pd.DataFrame([c], columns=file_naming)

	if total_profit_df.empty is True:
		total_profit_df = new_row
	else:
		total_profit_df = total_profit_df.append(new_row)
	total_profit_df.reset_index(inplace=True, drop=True)

