import pydicom
import pickle
import os
import numpy as np
import argparse

'''
Calculate alpha and beta for the fractionated LQ model
Parameters:
	k05 - The kill percentage at 0.5 Gy
	fracs - The number of fractions
Returns:
	alpha, beta
'''
def calc_alpha_beta_frac(k05, fracs):
	x1 = 5.
	x2 = 0.5
	k1 = 0.992
	beta = fracs / (x2 - x1) * (np.log(1 - k1) / x1 - np.log(1 - k05) / x2)
	alpha = -np.log(1 - k1) / x1 - beta * x1 / fracs
	return alpha, beta

'''
Calculate the raw blood cell kill using the fractionated LQ model
Parameters:
	counts, edges - A histogram of the blood matrix
	k05 - The kill percentage at 0.5 Gy
	fracs - The number of fractions
Returns:
	percent - The kill percentage
'''
def calc_kill_frac(counts, edges, k05=0.2044, fracs=5):
	#Raw kill
	alpha, beta = calc_alpha_beta_frac(k05, fracs)
	killed = np.sum(kill_contributions(counts, edges[:-1], alpha, beta, fracs))
	percent = killed / np.sum(counts)

	return percent

'''
Calculate the kill contributions for each dose range
Parameters:
	counts, edges - A histogram of the blood matrix
	alpha, beta - alpha and beta for the fractionated LQ model
	fracs - The number of fractions
Returns:
	The contribution to the total kill for each bin in the blood dose histogram
'''
def kill_contributions(counts, edges, alpha, beta, fracs):
	return  counts * (1. - np.exp(-fracs * edges * (alpha + beta * edges)))

'''
Apply regeneration to the percent kill figure
Parameters:
	percent - The raw kill percentage
	regen_rate - The regeneration rate in percentage points per day
	day - The measurement day (found that regeneration levels off at day 105)
Returns:
	The percent kill including regeneration
'''
def regeneration(percent, regen_rate, day=25):
	day = min(day, 105)
	percent -= (day - 25) * regen_rate if day > 25 else 0
	return percent

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('directory', type=str, help='The patient directory to look in')
	args = parser.parse_args()
	
	with open(os.path.join(args.directory, 'blood_hist.pickle'), 'rb') as infile:
		counts, edges = pickle.load(infile)
	percent = calc_kill_frac(counts, edges)
	percent = regeneration(percent, 0.001, day=25)
	print('Total Percent Kill:\t%g' % percent)
