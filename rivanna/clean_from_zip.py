# move to directory with zip files to run
import os
import shutil
from zipfile import ZipFile

# Set PARENT_PATH to be the directory in which each patient zip files are located
PARENT_PATH = '.'



def unzip(zippath):
    with ZipFile(zippath, 'r') as zipObj:
        zipObj.extractall()


def move_files_to_clinical(source, dst):
    files = os.listdir(source)
    print('source:', source)
    print('dst:',dst)
    
    for f in files:
        shutil.move(os.path.join(source,f), dst)


def rename_dcm(clinical_path):
    files = os.listdir(clinical_path)
    
    for f in files:
        header = f[:2]
        rest = f[2:]
        tail = f[-4:]
        new_name = ''
        if header == 'RS':
            new_name = 'RTSTRUCT' + rest
            if tail != '.dcm':
                new_name += '.dcm'
        elif header == 'RP':
            new_name = 'RTPLAN' + rest
            if tail != '.dcm':
                new_name += '.dcm'
        elif header == 'RD':
            new_name = 'RTDOSE' + rest
            if tail != '.dcm':
                new_name += '.dcm'
        elif header == 'CT' and tail != '.dcm':
            new_name = f + '.dcm'
            
        if new_name != '':
            old_path = os.path.join(clinical_path, f)
            new_path = os.path.join(clinical_path, new_name)
            os.rename(old_path, new_path)
        

for f in os.listdir(PARENT_PATH):
    if '.zip' in f:
        patient_name = f[:-4]
        zippath = os.path.join(PARENT_PATH,f)
        unzip(zippath)
        os.remove(zippath)
        clinical_path = os.path.join(PARENT_PATH, patient_name, 'clinical')
        
        try:
            os.mkdir(clinical_path)
        except:
            print('clinical dir already present')
            
        patient_path = os.path.join(PARENT_PATH, patient_name)
        
        move_files_to_clinical(patient_path, clinical_path)
        rename_dcm(clinical_path)
        
        csv_path = os.path.join(patient_path, 'csvs')
        try:
            os.mkdir(csv_path)
        except:
            print('csvdir already present')
        open(os.path.join(csv_path,'a.csv'),'a').close()
        
        





