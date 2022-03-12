import pydicom
import pickle
import argparse
import os
import numpy as np
import pandas as pd

from lymphkill.file_utils import find_prefixed_files, find_dicom_directory, load_rtdose_files, find_csv_directory
from lymphkill.static_organ_info import get_voxel_size

'''
organmask_dict - dict containing name, mask array, and other information about specific organ
doses - array containing doses to each voxel of the body
voxelsize - volume of each voxel (cc) 
returns boolean success value and pandas df containins coordinate, dose info
'''
def get_doses_to_organ(organmask_dict, doses, voxelsize):
	success = False
	organ = organmask_dict['Name']
	print('********getting doses to', organ,'**********')
	
	organ_dosemask = organmask_dict['Mask'] # mask array for organ - True if voxel in organ
	print('sum,shape - mask: ',np.sum(organ_dosemask), organ_dosemask.shape)
#	nonorgan_dosemask = ~organ_dosemask # mask array for outside organ - opposite of above
	
	#doses_to_organ = doses.copy()
	#doses_to_organ[nonorgan_dosemask] = 0 # array containing only doses to organ, all other 0

	doses_coord = [] # creates list of tuples. 0 is a tuple representing coord, 1 is dose 
	doses_nocoord = []
	doseshape = doses.shape # shape of dose files
	
	#loops through all voxels and adds voxels in organ to list	
	
	for axis0 in range(doseshape[0]):
		for axis1 in range(doseshape[1]):
			for axis2 in range(doseshape[2]):
				if organ_dosemask[axis0][axis1][axis2]:
					voxeldose = doses[axis0][axis1][axis2]
					doses_coord.append((axis0, axis1, axis2, voxeldose))
					doses_nocoord.append(voxeldose)
	
	if len(doses_nocoord) != 0:
		print('num doses to', organ, ': ', len(doses_nocoord))
		print('max dose to', organ, ': ', max(doses_nocoord))

		pd_cols = [
			'Axis0',
			'Axis1',
			'Axis2',
			"Dose"
		]	

		df = pd.DataFrame( columns=pd_cols, data=doses_coord )
		success = True
	else:
		df = None
		
	return success, df



parser = argparse.ArgumentParser()
parser.add_argument("directory", type=str, help='patient directory')
args = parser.parse_args()
# when running this file from within the patient directory pass argument .
# print(os.listdir(args.directory))
dcm_directory = find_dicom_directory(args.directory)
rtdose_files = find_prefixed_files(dcm_directory, 'RTDOSE')
csv_directory = find_csv_directory(args.directory)
dosegrids = load_rtdose_files(rtdose_files) 
#print(rtdose_files[0])


# 'rb' = read binary
#with open('contours.pickle', 'rb') as infile:
#	contours = pickle.load(infile)

with open(os.path.join(args.directory, 'masks.pickle'), 'rb') as infile:
	masks = pickle.load(infile)

voxelsize = get_voxel_size(rtdose_files[0])
#voxel volume
#dosegrids is a list of voxel doses in each file
#print(voxelsize)
#print(type(voxelsize))


doses = np.sum(np.array(dosegrids),axis=0)

print('shape of doses:', doses.shape)
print('voxelsize:', voxelsize)
failures = 0
failure_names = []
success = False

with open(os.path.join(args.directory, 'grid_shape.pickle'), 'wb') as outfile:
	pickle.dump(doses.shape, outfile)

with open(os.path.join(args.directory, 'voxelsize.pickle'), 'wb') as outfile:
	pickle.dump(voxelsize, outfile)

for mask in masks:
	organ = mask['Name'].lower()
	if ('heart' in organ) or ('aorta' in organ) or ('vc' in organ) or ('pa' in organ) or ('t5' in organ) or ('t10' in organ):
		success, df = get_doses_to_organ(mask, doses, voxelsize)
	else:
		success = False	

	if success:
		filename = organ + 'doses.csv'
		print('SUCCESS:', filename)
		df.to_csv(os.path.join(csv_directory, filename))
	else:
		failures += 1
		failure_names.append(organ)

print('Number of Failures:', failures)
for organ in failure_names:
	print(organ, 'Failed')



