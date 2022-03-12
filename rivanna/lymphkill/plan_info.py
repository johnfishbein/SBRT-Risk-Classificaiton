import pydicom
import argparse

from lymphkill.file_utils import find_dicom_directory, find_prefixed_file

'''
Get the beam information for a given plan
Parameters:
	directory - The patient directory to look for a plan in
Returns:
	total_mu, active_beams, time_per_beam: The number of MUs in the plan, the number of active beams, and the time on in seconds for each beam
'''
def get_beam_info(directory):
	dcm_directory = find_dicom_directory(directory)
	rtplan_file = find_prefixed_file(dcm_directory, 'RTPLAN')
	rtplan = pydicom.dcmread(rtplan_file)

	total_mu = 0
	active_beams = 0
	total_time = 0.
	for i, beam in enumerate(rtplan.FractionGroupSequence[0].ReferencedBeamSequence):
		total_mu += beam.BeamMeterset
		doserate = rtplan.BeamSequence[i].ControlPointSequence[0].DoseRateSet
		total_time += beam.BeamMeterset / doserate * 60
		if beam.BeamMeterset > 0:
			active_beams += 1	

	time_per_beam = total_time / active_beams
	
	return total_mu, active_beams, time_per_beam

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('directory', type=str, help='The patient directory to look in')
	args = parser.parse_args()

	total_mu, active_beams, time_per_beam = get_beam_info(args.directory)	
	print('Total MU: %d\nActive Beams: %d\nTime Per Beam: %g' % \
		(total_mu, active_beams, time_per_beam))
