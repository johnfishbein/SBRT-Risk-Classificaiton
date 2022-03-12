# Project Information

**Parsing Patient Data from dicom**
*Files in rivanna*
- run_pipeline.slurm
	- Runs the entire pipeline from start to finish on list of patients in a given patient directory specified in the file.
- lymphkill/structure_loading.py
	- Takes in path to patient directory as command line agrument. Parses patient dicom info and creates contours.pickle file saved in each specific patient directory
- lymphkill/mask_generation.py
	- Takes in path to patient directory as command line agrument. Uses contours.pickle file in patient directory to generate masks.pickle. Holds mask array for each contoured organ. 
- get_voxel_doses.py
	- Takes in path to patient directory as command line agrument. Uses masks.pickle along with RTDOSE dicom files to isolate organ specific doses. Saves doses to each specific organ in csvs subdirectory with name [organ]doses.csv. Also saves grid_shape.pickle and voxelsize.pickle in patient directory containing patient dosegrid shape and voxel size respectively.
- compute_statistics.py
	- Takes in path to patient directory as command line agrument. Uses patient grid_shape.pickle, voxelsize.pickle, and each csv file in csvs subdirectory to calculate dosimetry statistics for each organ.  Compiles statistics for all organs into a single dataframe and saves the output to patient_results.csv in patient directory.
- clean_from_zip.py
	- File has global variable PARENT_PATH.  This should contain path to directory which holds each patient's dicom info in a zip file named [patient initials].zip.  This file unzips each individual patient folder in the specified directory and creates the necessary directory structure required to run the entire pipeline.
- copy_directory_structure.py
	- This file copies only the patient_results.csv file generated from compute_statistics.py after the pipeline has been run on each patient. This file contains gloabal variables SOURCE_ROOT and DST_ROOT. SOURCE_ROOT should be the path to the data directory containing all of the individual patient directories.  DST_ROOT is the directory in which you want to save the results.  This goes through each patient directory and copies only the patient_results.csv file into a new subdirectory under the specified destination.

*Data Format*
- Each patient folder has two subdirectories:
	- clinical: Holds patient dicom files from Pinnacle.  These files must start with 1 of 4 prefixes: 'RTSTRUCT', 'RTDOSE', 'RTPLAN', 'CT, and each must end in the .dcm file ending. 
	csvs: Holds csv files containing doses specific to a given organ.  Before running, must contain empty file with .csv file ending so code can find the directory.


*Running Pipeline In Rivanna*
-	Given patient zip files, create directory that holds all of them.
- specify directory in clean_from_zip.py and run.
- In run_pipeline.slurm, switch email to your own, edit patients variable to match initials of your patients and edit PATIENT_DIR variab to match patient directory.
- run the command 'sbatch run_pipeline.slurm' to run all of the files necessary. 
- Each patient directory will have corresponding patient_results.csv.  Make a new results folder, and edit variables in and then run copy_directory_structure,py to isolate results.



**Running the classification**
*Files in local directory*
- label_data.csv
	- CSV file with data associated with each patient
- results directory
	- Directory with subdirectory for each patient.  Each patient subdirectory contains a single file: patient_results.csv
- data_cleaning.ipynb
	- Reads label_data.csv and each patient results file.  Cleans all of the data and compiles it together into a final dataframe with all of the results and patient info. Outputs this to classification_data.csv.
- classification_data.csv
	- Each row is a patient and each column is a feature about that patient.
- treatment_classification.ipynb
	- Performs Random Forest and SVC classification and analysis.

