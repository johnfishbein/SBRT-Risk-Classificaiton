#!/usr/bin/env python
# coding: utf-8



import os
import shutil

SOURCE_ROOT='./ProductionPatients/'
DST_ROOT='./results/'



os.listdir(SOURCE_ROOT)



for patient in os.listdir(SOURCE_ROOT):
    if len(patient) <=3:
        print('*****',patient,'*****')
        source_path = os.path.join(SOURCE_ROOT, patient)
        copied_path = os.path.join(DST_ROOT, patient)
        os.mkdir(copied_path)
        shutil.copyfile(os.path.join(source_path, 'patient_results.csv'), os.path.join(copied_path, 'patient_results.csv'))






