import platform

# Note that the code assumes that there is a trailing '/' on these
if platform.node().startswith('multilens'):
    detector_pickled_path = '/home/jacopo/sphere/pickled_detectors/'
    detector_calibration_path = '/home/jacopo/sphere/calibrations/'  # Path for the calibration files
    data_files_path = '/home/jacopo/sphere/'      # Path for the simulation data files
else:
    detector_pickled_path = '/chroma-data/configurations/'
    detector_calibration_path = '/chroma-data/configurations/'  # Path for the calibration files
    data_files_path = '/chroma-data/simulations/'               # Path for the simulation data files


def get_calibration_file_name_without_path(config,FV):
    return 'detresang-'+config+'_1DVariance_100million'+FV+'.root'

def get_calibration_file_name(config,FV=''):
    return detector_calibration_path+get_calibration_file_name_without_path(config,FV)

def get_data_file_path(config):
    return data_files_path+config+'/raw_data/'

    
