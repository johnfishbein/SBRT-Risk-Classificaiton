import pydicom
import pickle
import argparse
import os
import re
import numpy as np

from sys import exit

from lymphkill.calc_blood_dose import calc_blood_dose
from lymphkill.calc_blood_kill import calc_kill_frac
from lymphkill.file_utils import find_dicom_directory, find_prefixed_file, find_prefixed_files, load_rtdose_files
from lymphkill.mask_generation import mask_generation
from lymphkill.plan_info import get_beam_info
from lymphkill.static_organ_info import get_static_dose_info
from lymphkill.structure_loading import load_structures

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('directory', type=str)

	args = parser.parse_args()

	for i in os.listdir(args.directory):
		if 'zip' in i:
			cwd = os.getcwd()
			os.chdir(args.directory)
			os.system('unzip %s' % i)
			os.chdir(cwd)
			break
	
	dcm_directory = find_dicom_directory(args.directory)
	contours = load_structures(dcm_directory)

	with open(os.path.join(args.directory, 'contours.pickle'), 'wb') as outfile:
		pickle.dump(contours, outfile)
	
	try:
		ct_infos = [pydicom.dcmread(f) for f in find_prefixed_files(dcm_directory, 'CT')]
		dose_info = pydicom.dcmread(find_prefixed_file(dcm_directory, 'RTDOSE'))
	except Exception as ex:
		print(type(ex), ex)
		print('Could not load in ct/dose info')
		exit(0)
	
	masks = mask_generation(contours, ct_infos, dose_info)

	with open(os.path.join(args.directory, 'masks.pickle'), 'wb') as outfile:
		pickle.dump(masks, outfile)

	df = get_static_dose_info(args.directory)
	df.to_csv(os.path.join(args.directory, 'static_organ_dose.csv'))

	total_mu, active_beams, time_per_beam = get_beam_info(args.directory)
	print('Total MU: %d\nActive Beams: %d\nTime Per Beam: %g' % \
		(total_mu, active_beams, time_per_beam))
	
	rtdose_files = find_prefixed_files(dcm_directory, 'RTDOSE')
	dosegrids = load_rtdose_files(rtdose_files)
	blood_voxels = calc_blood_dose(masks, time_per_beam, dosegrids)

	counts, edges = np.histogram(blood_voxels,
		bins=np.arange(0, np.max(blood_voxels)+0.1, 0.1))
	
	with open(os.path.join(args.directory, 'blood_hist.pickle'), 'wb') as outfile:
		pickle.dump((counts, edges), outfile)
	with open(os.path.join(args.directory, 'blood_dose.pickle'), 'wb') as outfile:
		pickle.dump(blood_voxels, outfile)
	
	percent = calc_kill_frac(counts, edges)
	print('Total Percent Kill:\t%g' % percent)
