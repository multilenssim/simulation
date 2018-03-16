import platform

EXO_DATA_FILES = '/home/kwells/chroma-data/'
DEFAULT_CONFIG_DIR = 'configurations/'
DEFAULT_SIMULATION_DIR = 'simulations/'
JACOPO = False

sherlock_data_files = '/home/groups/gratta/kwells/chroma-data/'

# Note that the code assumes that there is a trailing '/' on these
if platform.node().startswith('exo3'):
    if JACOPO:
        detector_pickled_path = '/home/jacopodalmasson/Desktop/dev/sphere/pickled_detectors/'
        detector_calibration_path = '/home/jacopodalmasson/Desktop/dev/sphere/calibrations/'  # Path for the calibration files
        data_files_path = '/home/jacopodalmasson/Desktop/dev/sphere/'      # Path for the simulation data files
    else:
        detector_pickled_path = EXO_DATA_FILES+DEFAULT_CONFIG_DIR
        detector_calibration_path = detector_pickled_path               # Path for the calibration files
        data_files_path = EXO_DATA_FILES+DEFAULT_SIMULATION_DIR      	# Path for the simulation data files
else:
    detector_pickled_path = sherlock_data_files+DEFAULT_CONFIG_DIR
    detector_calibration_path = detector_pickled_path                   # Path for the calibration files
    data_files_path = sherlock_data_files+DEFAULT_SIMULATION_DIR        # Path for the simulation data files

# This is just for experimenting with hdf5 files
def get_calibration_file_name_base_without_path(config):
    return 'detresang-'+config+'_1DVariance_100million'

def get_calibration_file_name_without_path(config):
    return get_calibration_file_name_base_without_path(config)+'.root'

def get_calibration_file_name(config):
    return detector_calibration_path+get_calibration_file_name_without_path(config)

def get_data_file_path(config):
    return data_files_path+config+'/raw_data/'

def get_data_file_path_no_raw(config):
    return data_files_path+config+'/'
