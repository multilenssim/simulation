import platform

EXO_DATA_DIR = '/home/jacopo/chroma-data/'
DEFAULT_CONFIG_DIR = 'configurations/'
DEFAULT_SIMULATION_DIR = 'simulations/'
JACOPO = False

SHERLOCK_DATA_FILES = '/home/groups/gratta/kwells/chroma-data/'
AWS_DATA_FILES = '/chroma-data/'

# Note that the code assumes that there is a trailing '/' on these
if platform.node().startswith('exo3'):
    if JACOPO:
        detector_config_path = '/home/jacopodalmasson/Desktop/dev/sphere/pickled_detectors/'
        detector_calibration_path = '/home/jacopodalmasson/Desktop/dev/sphere/calibrations/'  # Path for the calibration files
        data_files_path = '/home/jacopodalmasson/Desktop/dev/sphere/'      # Path for the simulation data files
    else:
        detector_config_path = EXO_DATA_DIR+DEFAULT_CONFIG_DIR
        detector_calibration_path = detector_config_path               # Path for the calibration files
        data_files_path = EXO_DATA_DIR+DEFAULT_SIMULATION_DIR        # Path for the simulation data files
elif platform.node().startswith('multilens'):
    detector_config_path = EXO_DATA_DIR+DEFAULT_CONFIG_DIR
    detector_calibration_path = detector_config_path               # Path for the calibration files
    data_files_path = EXO_DATA_DIR+DEFAULT_SIMULATION_DIR        # Path for the simulation data files
else:
    detector_config_path = AWS_DATA_FILES+DEFAULT_CONFIG_DIR
    detector_calibration_path = detector_config_path                   # Path for the calibration files
    data_files_path = AWS_DATA_FILES+DEFAULT_SIMULATION_DIR            # Path for the simulation data files

def get_calibration_file_name_base_without_path(config,FV=''):               # To enable switching between ROOT and hdf5 files easily
    return 'detresang-'+config+'_1DVariance_100million'

def get_calibration_file_name_without_path(config,FV=''):
    return get_calibration_file_name_base_without_path(config,FV)+'.h5'

def get_calibration_file_name(config,FV=''):
    return detector_calibration_path+get_calibration_file_name_without_path(config,FV)

def get_data_file_path(config):
    return data_files_path+config+'/raw_data/'

def get_data_file_path_no_raw(config):
    return data_files_path+config+'/'
