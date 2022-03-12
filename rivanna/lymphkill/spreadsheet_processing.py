import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

'''
Get a list of all useful row subsets of a given dataframe
Parameters:
	df - The dataframe
Returns:
	dct - The dictionary of row subsets
		All - df
		Arc - df['Plan type'] == 'arc'
		IMRT - df['Plan type'] == 'imrt'
		Gated - df['Gated'] == 'y'
		Non-Gated - df['Gated'] == 'n'
		813 - df['RTOG'] == '813'
		915 - df['RTOG'] == '915'
		DT < 200 - df['Total Time'] < 200
		DT 200-300 - df['Total Time'] in [200, 300)
		DT > 300 - df['Total Time'] >= 300
		TV < 20 - df['Tumor Volume (cc)'] < 20
		TV 20-40 - df['Tumor Volume (cc)'] in [20, 40)
		TV 40-60 - df['Tumor Volume (cc)'] in [40, 60)
		TV > 60 - df['Tumor Volume (cc)'] >= 60
'''
def get_all_subsets(df):
	dct = {'All': df}
	dct['Arc'] = df.loc[df['Plan type'] == 'arc']
	dct['IMRT'] = df.loc[df['Plan type'] == 'imrt']
	dct['Gated'] = df.loc[df['Gated'] == 'y']
	dct['Non-Gated'] = df.loc[df['Gated'] == 'n']
	dct['813'] = df.loc[df['RTOG'] == 813]
	dct['915'] = df.loc[df['RTOG'] == 915]
	dct['DT < 200'] = df.loc[df['Total Time'] < 200]
	dct['DT 200-300'] = df.loc[(df['Total Time'] >=  200) & (df['Total Time'] < 300)]
	dct['DT > 300'] = df.loc[df['Total Time'] >= 300]
	dct['TV < 20'] = df.loc[df['Tumor Volume (cc)'] < 20]
	dct['TV 20-40'] = df.loc[(df['Tumor Volume (cc)'] >= 20) & (df['Tumor Volume (cc)'] < 40)]
	dct['TV 40-60'] = df.loc[(df['Tumor Volume (cc)'] >= 40) & (df['Tumor Volume (cc)'] < 60)]
	dct['TV > 60'] = df.loc[df['Tumor Volume (cc)'] >= 60]
	return dct

'''
Print the average blood fraction receiving each dose
Parameters:
	df - The dataframe
'''
def get_average_fractions(df):
	frac_cols = [col for col in df.columns if 'Gy' in col]
	means = df[frac_cols].mean()

	outstr1 = ''
	outstr2 = ''
	for col in frac_cols:
		outstr1 += '%10s ' % col
		outstr2 += '%10g ' % means[col]
	print(outstr1)
	print(outstr2)

'''
Print accuracy metrics (absolute LYA difference, percentage difference, min diff, max diff, num < 0.1, 0.3, 0.5) for the dataframe
Parameters:
	df - The dataframe
'''
def get_accuracy_info(df):
	post_tx = {}
	post_tx['flq'] = df['Pre-Tx LYA'] * (1 - df['Predicted KP (FLQ)'])
	post_tx['lq'] = df['Pre-Tx LYA'] * (1 - df['Predicted KP (LQ)'])
	post_tx['lin'] = df['Pre-Tx LYA'] * (1 - df['Predicted KP (Lin)'])
	post_tx['nak'] = df['Pre-Tx LYA'] * (1 - df['Predicted KP (Nak)'])

	print('%8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s' % ('Func', 'Abs Diff', '% Diff', 'Min', 'Max', 'D<0.1', 'D<0.3', 'D<0.5'))
	for key in post_tx:
		diff = np.abs(post_tx[key] - df['Post-Tx LYA'])
		abs_diff = np.mean(diff)
		perc_diff = np.mean(diff / df['Pre-Tx LYA']) * 100
		min_diff = np.min(diff)
		max_diff = np.max(diff)
		nl01 = np.sum(diff <= 0.1)
		nl03 = np.sum(diff <= 0.3)
		nl05 = np.sum(diff <= 0.5)
		print('%8s  %8g  %8g  %8g  %8g  %8g  %8g  %8g' % (key, abs_diff, perc_diff, min_diff, max_diff, nl01, nl03, nl05))

'''
Generate a plot of LYA difference (sim vs. measured) vs Pre-Tx LYA
Parameters:
	dct - The dictionary of subsets from get_all_subsets
	subsets - The subset keys to plot
'''
def get_accuracy_plot(dct, subsets):
	fig = plt.figure()
	max_x = 0
	for subset in subsets:
		df = dct[subset]
		post_tx = df['Pre-Tx LYA'] * (1 - df['Predicted KP (FLQ)'])
		diff = post_tx - df['Post-Tx LYA']
		max_x = np.max(df['Pre-Tx LYA'])
		error = df['Pre-Tx LYA'] * df['Error (FLQ)']
		plt.errorbar(df['Pre-Tx LYA'], diff, yerr=error, fmt='o', label=subset)
	plt.plot([0, max_x*1.1], [0, 0], color='black')
	plt.xlim([0, max_x*1.1])
	plt.xlabel(r'Pre-Treatment LYA (cells/L x $10^9$)')
	plt.ylabel(r'LYA difference (Simulated vs. Measured) (cells/L x $10^9$')
	if len(subsets) > 1:
		plt.legend()
	plt.show()

'''
The spreadsheet has the following columns:
	Patient - The patient initials
	Plan type - arc or imrt
	Beams - Number of beams
	Total Time - Total beam on time (not including gating)
	Gated - y/n
	RTOG - 813 (central) or 915 (peripheral)
	Tumor Volume (cc)
	Measurement Day
	Patient Age
	Pre-Tx LYA
	Post-Tx LYA
	Predicted KP (FLQ) - Predicted kill percentage (Fractionated LQ)
	Error (FLQ)
	Predicted KP (LQ) - Predicted kill percentage (LQ)
	Error (LQ)
	Predicted KP (Lin) - Predicted kill percentage (Linear Spline)
	Error (Lin)
	Predicted KP (Nak) - Predicted kill percentage (Nakamura best fit)
	< 0.5 Gy
	0.5-0.6 Gy
	0.6-0.7 Gy
	0.7-0.8 Gy
	0.8-0.9 Gy
	0.9-1.0 Gy
	1.0-2.0 Gy
	2.0-3.0 Gy
	> 3.0 Gy
'''
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('spreadsheet', type=str, help='Path to spreadsheet')
	args = parser.parse_args()

	df = pd.read_csv(args.spreadsheet)
	df['RTOG'] = pd.to_numeric(df['RTOG'], errors='coerce')
	df['Tumor Volume (cc)'] = pd.to_numeric(df['Tumor Volume (cc)'], errors='coerce')

	subsets = get_all_subsets(df)
	for subset in subsets:
		print('%s\tN=%d' % (subset, len(subsets[subset])))
		get_average_fractions(subsets[subset])
		get_accuracy_info(subsets[subset])
	get_accuracy_plot(subsets, ['All'])
	get_accuracy_plot(subsets, [s for s in subsets if 'Gated' in s])
	get_accuracy_plot(subsets, [s for s in subsets if 'TV' in s])
	get_accuracy_plot(subsets, [s for s in subsets if 'DT' in s])
