import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pydicom

'''
Find the first file with a given prefix in a directory
Parameters:
	directory - The directory to look in
	prefix - The file prefix to look for
Returns:
	The name of the file, or None if none were found
'''
def find_prefixed_file(directory, prefix):
	for i in os.listdir(directory):
		if re.search('^'+prefix, i) is not None:
			return os.path.join(directory, i)
	return None

'''
Find the all files with a given prefix in a directory
Parameters:
	directory - The directory to look in
	prefix - The file prefix to look for
Returns:
	The names of the files, or None if none were found
'''
def find_prefixed_files(directory, prefix):
	files = []
	for i in os.listdir(directory):
		if re.search('^'+prefix, i) is not None:
			files.append(os.path.join(directory, i))
	return files

'''
Finds all files with given postfix in specified directory
Parameters:
	directory - directory to look in
	postfix - postfix of file to look for
Returns:
	List of all names of files with desired postfix
'''
def find_postfixed_files(directory, postfix):
	files = []
	postfix_len = len(postfix)
	for f in os.listdir(directory):
		if f[-1*postfix_len:] == postfix:
			files.append(os.path.join(directory, f))
	return files

'''
Find the subdirectory containing dicom files
Parameters:
	directory - The directory to look in
Returns:
	The name of the directory, or None if none was found
'''
def find_dicom_directory(directory):
	for i in os.listdir(directory):
		subd = os.path.join(directory, i)
		if os.path.isdir(subd):
			for j in os.listdir(subd):
				if j[-4:] == '.dcm':
					return subd
	return None
'''
Find the subdirectory containing csv files
Parameters:
	directory - directory to look in
Returns:
	The name of the directory, or None if none were found
'''
def find_csv_directory(directory):
	for i in os.listdir(directory):
		subd = os.path.join(directory, i)
		if os.path.isdir(subd):
			for j in os.listdir(subd):
				if j[-4:] == '.csv':
					return subd
	return None

'''
Load the dose grids from the RTDOSE files
Parameters:
	files - A list of rtdose files
Returns:
	A list of dose grids
'''
def load_rtdose_files(files):
	dosegrids = []
	for i, fname in enumerate(files):
		data = pydicom.dcmread(fname)
		raw = (data.pixel_array * data.DoseGridScaling).astype(float)
		if np.max(raw) == 0:
			continue
		dosegrids.append(raw.transpose(1, 2, 0))

	return dosegrids

'''
Play a video of a sequence of image frames
Parameters:
	Cube - an X x Y x Z ndarray, where Z is the number of frames
'''
def implay(cube):
	plt.ion()
	plt.figure()
	plt.show()

	for i in range(cube.shape[2]):
		plt.clf()
		plt.title('Frame %d' % (i+1))
		plt.imshow(cube[:, :, i])
		plt.colorbar()
		plt.pause(0.01)
	
	plt.ioff()
