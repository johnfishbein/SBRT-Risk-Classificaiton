#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --output=run_pipeline.out
#SBATCH --error=run_pipeline.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jhf5my@virginia.edu
#SBATCH --account=phys_nrf
#SBATCH --partition=standard
#SBATCH --mem=30000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --tasks=1

module unload python
module load gcc/7.1.0
module load intel/18.0
module load mvapich2/2.3.1
module load openmpi/3.1.4
module load python/3.6.6

source project_venv/bin/activate

#extra=("WP/" "BJ/")


#14
#patients=("BB2/" "BC/" "CT/" "DE/" "DM/" "FJ/" "GB/" "HI/" "MB/" "PG/" "PP/" "RD/" "SB/" "SR1/")

#9
#patients=("AJ/" "BL/" "BY/" "DA/" "MJ1/" "PJ/" "PM/" "SW/" "TA/")

#11
#patients=("DJ/" "FJ1/" "GV/" "HD/" "HL/" "IB/" "IP/" "KS/" "LK/" "MK/" "VE/")

#12
#patients=("BE" "CW" "DJ1" "HM1" "LJ" "LL" "MM1" "PJ2" "TC" "WJ" "WL" "WT") 

patients=("BB2/" "BC/" "CT/" "DE/" "DM/" "FJ/" "GB/" "HI/" "MB/" "PG/" "PP/" "RD/" "SB/" "SR1/" "AJ/" "BL/" "BY/" "DA/" "MJ1/" "PJ/" "PM/" "SW/" "TA/" "DJ/" "FJ1/" "GV/" "HD/" "HL/" "IB/" "IP/" "KS/" "LK/" "MK/" "VE/" "BE" "CW" "DJ1" "HM1" "LJ" "LL" "MM1" "PJ2" "TC" "WJ" "WL" "WT") 

PATIENT_DIR="ProductionPatients"

for t in ${patients[@]}; do
	echo Running Patient $t at path: $PATIENT_DIR/$t
	python lymphkill/structure_loading.py $PATIENT_DIR/$t
	python lymphkill/mask_generation.py  $PATIENT_DIR/$t
	python get_voxel_doses.py $PATIENT_DIR/$t
	rm $PATIENT_DIR/$t/contours.pickle
	python compute_statistics.py $PATIENT_DIR/$t
done


