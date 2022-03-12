#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
import math
import argparse


def dist(p1, p2, vox_sidelen):
    '''returns euclidian distance between p1 and p2 in dimentions of sidelen'''
    
    sqr = math.sqrt( (p1[0] - p2[0])**2  + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2 )
    return sqr * vox_sidelen


def get_all_organ_dfs(patient_path, masks):
    '''returns organs dict of organ where organs[organ][organDF] is dose dataframe'''
    '''organs[organ][organMask] is dose mask of organ'''
    '''organs[organ][dosegrid] is dosegrid of organ'''
    organs = {}
    for file in os.listdir(patient_path):
        if file[-4:] == '.csv' and len(file) != 5 and 'doses' in file:
            #print(file)
            df = pd.read_csv(os.path.join(patient_path, file), index_col=0)
            
            # gets organ name from filename
            organ = file[:-9]
            # sets dose/coordinate dataframe for organ
            print('Getting DF and Mask for', organ)
            organs[organ] = {'organDF': df}
            
            # sets mask array for organ
            for mask in masks:
                name = mask['Name'].lower()
                if name == organ:
                    organs[organ]['organMask'] = mask['Mask'] 
            
    return organs



def in_grid(mask_shape, ax0, ax1, ax2):
    if ax0 < 1 or ax1 < 1 or ax2 < 1:
        return False
    if ax0 >= mask_shape[0]-1 or ax1 >= mask_shape[1]-1 or ax2 >= mask_shape[2]-1:
        return False
    
    return True
        

def close(coord, mask):
    '''helper function for get_nearby_high_doses'''
    '''returns true if given coordinate is within 1 voxel of organ'''
    
    mask_dim = mask.shape
    near = False
    ax0, ax1, ax2 = coord
    
    
    if not in_grid(mask_dim, ax0, ax1, ax2):
        return False
    try:
        if mask[ax0+1][ax1][ax2] or mask[ax0+1][ax1+1][ax2] or mask[ax0+1][ax1][ax2+1] or mask[ax0+1][ax1-1][ax2] or mask[ax0+1][ax1][ax2-1]:# or mask[ax0+1][ax1+1][ax2+1] or mask[ax0+1][ax1-1][ax2-1]:
            near = True
        if mask[ax0-1][ax1][ax2] or mask[ax0-1][ax1][ax2+1] or mask[ax0-1][ax1-1][ax2] or mask[ax0-1][ax1][ax2-1]: #or mask[ax0-1][ax1+1][ax2+1] or mask[ax0-1][ax1-1][ax2-1]:
            near = True
        if mask[ax0][ax1+1][ax2] or mask[ax0][ax1+1][ax2+1] or mask[ax0][ax1][ax2+1] or mask[ax0][ax1-1][ax2] or mask[ax0][ax1-1][ax2-1] or mask[ax0][ax1][ax2-1]:
            near = True
    except:
        print('Error at coordinates:', ax0,ax1,ax2)
    
    return near



def get_nearby_high_doses(organs):
    '''takes in all organ dicts and adds doses that are cutoff from each organ'''
    otherDF = organs['otherorgans']['organDF']
    cols = ['Axis0', 'Axis1', 'Axis2', 'Dose']
    for organ in organs:
        print('getting nearby high doses', organ)
        # other organs contains edge doses so skip that 'organ'
        if organ in 'otherorgans t5 t10':
            continue
        
        # dose/coordinate dataframe for organ
        organDF = organs[organ]['organDF']
        # current max dose to that organ
        cur_organ_max = organDF['Dose'].max()
        # dose mask of organ
        cur_organ_mask = organs[organ]['organMask']
        # copy of dose/coordinate datafram for OtherOrgans
        check_nearby = otherDF.copy()
        
        
        # creates dose/coordinate dataframe of doses to OtherOrgans that are within 10Gy of the max current organ
        check_nearby = check_nearby[check_nearby.Dose > (cur_organ_max - 10)]
        check_nearby = check_nearby[check_nearby.Dose < (cur_organ_max + 10)]
        
        # loops through each dose/coordinate in OtherOrgans that could be significant
        new_doses = []
        for row in check_nearby.itertuples():
            coord = row[1:4]
            if close(coord, cur_organ_mask):
                new_doses.append(row[1:])
            
        new_dosesDF = pd.DataFrame(new_doses, columns=cols)
        organs[organ]['organDF'] = pd.concat([organDF, new_dosesDF], ignore_index=True)
        print(len(new_doses), 'new doses to', organ)
    


def get_organ_max(organ, patient_path):
    '''takes in dict for individual organ and sets max dose and coord'''
    
    df = organ['organDF']
    organ_max = df[df['Dose'] == df['Dose'].max()]
    max_coord = [int(organ_max.Axis0.iloc[0]), int(organ_max.Axis1.iloc[0]), int(organ_max.Axis2.iloc[0])]
    organ['max_coord'] = max_coord
    organ['max_dose'] = float(df['Dose'].max())
    #print(organ['max_dose'])
    

def get_volume(organ, vox_vol):
    '''takes in organ dict and sets volume'''
    num_voxels = np.count_nonzero(organ['organMask'])
    organ['volume'] = num_voxels * vox_vol
    


# dose checks set to [25] -> different X vals
def get_VXcc(organ, vox_vol, X):
    '''takes in dict for organ, sets VX and VX mean coord'''
    '''given X, VX is the volume of organ recieving at least X dose'''
    
    df = organ['organDF']
    # new df with only dose/coordinate where dose >= X
    vX = df[df.Dose >= X]
    key1 = 'v' + str(X) + 'cc'
    
    #counts desired doses to determine vol
    organ[key1] = vX.Dose.count()*vox_vol
    
    # finds mean of desired coordinates
    mean_coord = [ vX.Axis0.mean(), vX.Axis1.mean(), vX.Axis2.mean() ]
    key2 = 'v' + str(X) + '_mean_coord'
    organ[key2] = mean_coord

    '''
    - not sure how relevant mean coordinate will be as doses will be spread out
    - assuming areas recieving high doses will be in the same general area
    - in the event of high dose 'islands' this is irrelevant
    '''
     


def dist_ofmax_from_T10(organ, t10_coord, vox_sidelen):
    '''assuming T10 is contoured, takes in organ dict, coordinates of t10, and voxel dimension'''
    '''sets distance of max dose from t10 since we set t10 to be origin in new coordinate system'''
    
    coord = organ['max_coord']
    distance = dist(coord, t10_coord, vox_sidelen)
    organ['dist_to_max_from_t10'] = distance



def normalize_max_coord(organ, t10_coord, t5_coord, dosegrid_shape):
    t5 = np.asarray(t5_coord)
    t10 = np.asarray(t10_coord)
    max_coord = np.asarray(organ['max_coord'])
    
    norm = max_coord - t10
    shape = np.asarray(dosegrid_shape)
    norm = norm/shape
    organ['normalized_max_coord'] = norm
    organ['axial_scale_factors'] = dosegrid_shape
    
    

# volume checks set to [0.5, 1, 4] -> different X vals
# not used anymore, replaced with building 
def get_DXcc(organ, vox_vol, X):
    '''takes in organ dict, X volume value, and voxel volume determine DXcc'''
    '''given X, DXcc is the highest dose recieved by Xcc of organ'''
    '''equivalently, DXcc is the minimum dose to the highest Xcc voxels'''
    
    key = 'D' + str(X) + 'cc'
    
    # calculates number of voxels to make Xcc -> Xcc/(cc/voxel)
    num_voxels = int(( X / vox_vol ))
    organDF = organ['organDF']
    # gets number of voxels in organ
    highest_index = len(organDF)-1
    

    if highest_index >= num_voxels:
        # sorts doses in increasing order and takes the lowest dose to the highest (num_voxels) doses
        dval = organDF.sort_values('Dose').iloc[(highest_index - num_voxels)].Dose
    else:  # if Xcc of organ does not receive a dose, DXcc is the min dose to that organ
        dval = organDF.Dose.min()
    
    organ[key] = dval
    

def build_DVH(organ_name, organ, vox_volume, DXcc_checks, save='n', savepath=''):
	step = 0.01
	doses = np.array(organ['organDF'].Dose)
	
	#D_x is array holding x axis values (Dose)
	#y is array holding number of voxels receiving each dose increment
	D_x = np.arange(0, np.max(doses)+5, step)
	y = np.zeros(D_x.shape)

	for dose in doses:
		# gets max index in D_x corresponding to dose and adds 1 to each y below it
		# effectivley tracks number of voxles recivieng at least given dose
		max_indx = int(dose*(1/step))
		for i in range(max_indx+1):
			y[i] += 1

	# y transfered into volume recieving each dose
	y = y*vox_volume
	plt.figure()
	plt.title('Cumulative DVH: ' + organ_name.upper())
	plt.xlabel('Dose(Gy)')
	plt.ylabel('Volume Receiving XDose')

	# sets ticks on y axis depending on volume

	if np.max(y) > 50:
		step_y = 10
	elif np.max(y) > 30:
		step_y = 5
	elif np.max(y) > 10:
		step_y = 2
	elif np.max(y) < 1:
		step_y = 0.5
	else:
		step_y = 1
	
	plt.yticks(np.arange(0, max(np.max(y),5), step_y))
	plt.plot(D_x, y)
	z = np.ones(D_x.shape)

	leg = ['DVH']
	for x in DXcc_checks:
		# sum(y>=x) holds number of indicies in y with volume higher than x cc
		# equivalently, holds index recieving x cc of dose
		# multiplying by step yields actual dose 
		DXcc = sum(y>=x)*step
		key = 'D' + str(x) + 'cc'
		organ[key] = DXcc
		leg.append(key + ': ' + str(DXcc))
		# z is all 1s, times x is a horizontal line at y=x
		plt.plot(D_x, z*x)

	plt.legend(leg)
	if save == 'y':
		figname = organ_name + '_DVH.png'
		plt.savefig(os.path.join(savepath, figname))

def NTCP(X, g, D50):
    return np.exp(X)/(1 + np.exp(X))

def get_NTCP_organ(organ, DXchecks):
    gd50 = {'V25': [-0.0235949, 5.796806],
            'D4': [-0.07193553, 6.95422],
            'D1': [-0.1503416, 11.03224],
            'D0.5': [-0.1588655, 11.86569],
            'Dmax': [-0.1550583, 12.55338],
           }

    organ['Dmax_NTCP'] = NTCP(organ['max_dose'], gd50['Dmax'][0], gd50['Dmax'][1])
    organ['V25_NTCP'] = NTCP(organ['v25cc'], gd50['V25'][0], gd50['V25'][1])

    for X in DXchecks:
        key = 'D' + str(X)
        organ[key + '_NTCP'] = NTCP(organ[key + 'cc'], gd50[key][0], gd50[key][1])



def write_to_csv(patient_organs, patient_path, write, patient_name):
	# index specifies which rows to keep for each organ
    df = pd.DataFrame([], index=['max_dose', 'max_coord', 'normalized_max_coord',
                                 'axial_scale_factors', 'v25cc', 'D0.5cc', 'D1cc', 
                                 'D4cc', 'dist_to_max_from_t10', 'voxel_volume',
                                 'Dmax_NTCP', 'V25_NTCP', 'D0.5_NTCP', 'D1_NTCP', 'D4_NTCP'])
    
    for organ in patient_organs:
        if organ in 't5t10' or organ == 'otherorgans':
            continue
        
        o = organ.lower()
        if 'heart' in o or 'vc' in o or 'aorta' in o or 'pa' in o:
            print('compiling', organ)
            organDF = pd.DataFrame.from_dict(patient_organs[organ], orient='index')
            df[organ] = organDF
    
    df.index.name = patient_name
    
    if write == 'y':
        df.to_csv(os.path.join(patient_path, 'patient_results.csv'))
        print('patient data written to patient_results.csv')
    else:
        print('results not written to file')
    return df
    
    



def pipeline(patient_path, patient_name):
    with open(os.path.join(patient_path, 'masks.pickle'), 'rb') as maskfile:
        masks = pickle.load(maskfile)

    patient_organs = get_all_organ_dfs(os.path.join(patient_path, 'csvs/'), masks)
    #get_nearby_high_doses(patient_organs)  # not used

    with open(os.path.join(patient_path, 'voxelsize.pickle'), 'rb') as infile:
        vox_vol = pickle.load(infile)

    #vox_vol = 0.015625 #cc = cm^3
    vox_sidelen = vox_vol**(1/3) 
	#vox_sidelen = 0.25 #cm = voxel_size^(1/3)
    VXcc_checks = [25]
    DXcc_checks = [0.5, 1, 4]
    
    contours_pres = False  # variable to determine whether t5/t10 are present
    if 't5' in patient_organs:
        dfT5 = patient_organs['t5']['organDF']
        dfT10 = patient_organs['t10']['organDF']
        t5meanpt = [dfT5.Axis0.mean(), dfT5.Axis1.mean(), dfT5.Axis2.mean()] 
        t10meanpt = [dfT10.Axis0.mean(), dfT10.Axis1.mean(), dfT10.Axis2.mean()]
        contours_pres = True
    
    
    for organ_name in patient_organs:
        #print('getting organ info for', organ_name)
        organ = patient_organs[organ_name]
        organ['voxel_volume'] = vox_vol
        #print('Getting max dose...')
        get_organ_max(organ, patient_path)
        #print('Getting Volume...')
        #get_volume(organ, vox_vol)
        
        if organ_name not in 'otherorgans t5 t10':
            #print('Getting VXcc...')
            for X in VXcc_checks:
                get_VXcc(organ, vox_vol, X)
            #print('Getting DXcc...')
            build_DVH(organ_name, organ, vox_vol, DXcc_checks, save='n', savepath=patient_path)
            get_NTCP_organ(organ, DXcc_checks)
#			for X in DXcc_checks:
#                get_DXcc(organ, vox_vol, X)
        
        if contours_pres:
            print('Getting distance between max dose and T10')
            dist_ofmax_from_T10(organ, t10meanpt, vox_sidelen)
            
            with open(os.path.join(patient_path, 'grid_shape.pickle'), 'rb') as infile:
                shape = pickle.load(infile)
            print('Normalizing max coord')
            #shape = [110,140,123]
            normalize_max_coord(organ, t10meanpt, t5meanpt, shape)
    
    message = 'Write results for patient ' + patient_name + ' to csv file? '
    #write = input(message)
    write = 'y'
    final_df = write_to_csv(patient_organs, patient_path, write, patient_name)
    
    
    return patient_organs, final_df
    
    

parser = argparse.ArgumentParser()
parser.add_argument('patient_directory', type=str)
args = parser.parse_args()

print('Pipeline: patient directory', args.patient_directory)
patient_name = args.patient_directory.strip('/')[args.patient_directory.strip('/').rfind('/')+1:]
#print(patient_name)
pipeline(args.patient_directory, patient_name)

  

