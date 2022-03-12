import pydicom
import numpy as np
import pickle
import argparse
import os
import pandas as pd

from lymphkill.file_utils import find_prefixed_files, find_dicom_directory, load_rtdose_files, implay
from lymphkill.mask_generation import print_mask
'''
Get the volume of a voxel
Parameters:
	fname - The dicom file to look in
Returns:
	voxelsize - The volume of the voxel
'''
def get_voxel_size(fname):
	data = pydicom.dcmread(fname)
	voxelsize = data.PixelSpacing[0] * data.PixelSpacing[1] * data.SliceThickness
	voxelsize /= 1000. #Rescale from mm^3 to cm^3
	return voxelsize

'''
Give static dose information for an organ
Parameters:
	mask - The mask structure from mask_generation.py for this organ
	dosegrid - The dose grid to check for
	voxelsize - The volume of a single voxel
	dosechecks - Dose thresholds to check at
Returns:
	maxdose, meandose, volume, intdose, dosevols
	maxdose - Maximum dose in the organ
	meandose - Average dose in the organ
	volume - Total volume of the organ
	intdose - Integral dose for the organ
	dosevols - Volumes for each dose threshold
'''
def get_organ_info(mask, dosegrid, voxelsize, dosechecks=[5, 10, 15, 20]):
	dosemask = dosegrid[mask['Mask']]
	print('Organ : ', mask['Name'], dosemask.shape)
	if mask['Name'] == 'aorta_CA':
		for voxel in dosemask:
			print(voxel)
	


	print('\n')
	maxdose = np.max(dosemask)
	meandose = np.sum(dosemask) / np.sum(mask['Mask'])
	volume = np.sum(mask['Mask']) * voxelsize
	intdose = meandose * volume
	dosevols = [np.sum(dosemask >= j) * voxelsize for j in dosechecks]

	return maxdose, meandose, volume, intdose, dosevols

'''
Get static dose information for a patient's organs
Parameters:
	masks - The mask structures from mask_generation.py for the organs to check
	dosegrid - The dose grid to check for
	voxelsize - The volume of a single voxel
	dosechecks - Dose thresholds to check at
Return:
	A pandas dataframe containing all of the information
'''
def get_organs_dataframe(masks, dosegrid, voxelsize, dosechecks=[5, 10, 15, 20]):
	print('shape of dosegrid', dosegrid.shape)
	print()
	data = []
	for mask in masks:
		print_mask(mask)
		
		mx, mn, vl, itd, dv = get_organ_info(mask, dosegrid, voxelsize, dosechecks)
		data.append({'Organ': mask['Name'], 'MaxDose': mx, 'MeanDose': mn, 
					 'TotalVolume (cm^3)': vl, 'IntegralDose': itd})
		for i in range(len(dosechecks)):
			data[-1]['V%d dosevol' % dosechecks[i]] = dv[i]
	data.append({'Organ': 'ALL', 
				 'MaxDose': np.max(dosegrid), 
				 'MeanDose': np.sum(dosegrid) / np.sum(dosegrid.astype(bool)),
				 'TotalVolume (cm^3)': np.sum(dosegrid.astype(bool)) * voxelsize,
				 'IntegralDose': np.sum(dosegrid) * voxelsize})
	for i in dosechecks:
		data[-1]['V%d dosevol' % i] = np.sum(dosegrid >= i) * voxelsize
	df = pd.DataFrame(data)

	for i in dosechecks:
		df['V%d dosevol/volume' % i] = df['V%d dosevol' % i] / df['TotalVolume (cm^3)']
	
	return df

'''
Load information from a patient directory and get a dataframe of organ information
Parameters:
	directory - The directory to look in
	patientname - The name of the patient. None if it is the name of the directory
	dosechecks - The dose volume thresholds to check at
'''
def get_static_dose_info(directory, patientname=None, dosechecks=[5, 10, 15, 20]):
	dcm_directory = find_dicom_directory(directory)
	rtdose_files = find_prefixed_files(dcm_directory, 'RTDOSE')
	dosegrids = load_rtdose_files(rtdose_files)
	voxelsize = get_voxel_size(rtdose_files[0])
	
	with open(os.path.join(directory, 'masks.pickle'), 'rb') as infile:
		masks = pickle.load(infile)
	
	df = get_organs_dataframe(masks, np.sum(np.array(dosegrids), axis=0), voxelsize, dosechecks)
		
	#Add in patient name
	if patientname is None:
		toks = directory.split('/')
		df['Patient'] = toks[-1] if len(toks[-1]) > 0 else toks[-2]
	else:
		df['Patient'] = patientname

	#Reorder columns
	cols = ['Patient', 'Organ', 'MaxDose', 'MeanDose', 'TotalVolume (cm^3)', 'IntegralDose']
	for i in dosechecks:
		cols.append('V%d dosevol' % i)
		cols.append('V%d dosevol/volume' % i)
	df = df[cols]

	return df

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('directory', type=str, help='The patient directory to look in')
	args = parser.parse_args()

	df = get_static_dose_info(args.directory)
	print(df)
	df.to_csv(os.path.join(args.directory, 'static_organ_dose.csv'))



