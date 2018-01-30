import platform

exo_data_files = '/home/kwells/chroma_data/'
JACOPO = False

# Note that the code assumes that there is a trailing '/' on these
if platform.node().startswith('exo3'):
    if JACOPO:
        detector_pickled_path = '/home/jacopodalmasson/Desktop/dev/sphere/pickled_detectors/'
        detector_calibration_path = '/home/jacopodalmasson/Desktop/dev/sphere/calibrations/'  # Path for the calibration files
        data_files_path = '/home/jacopodalmasson/Desktop/dev/sphere/'      # Path for the simulation data files
    else:
        detector_pickled_path = exo_data_files+'configurations/'
        detector_calibration_path = exo_data_files+'configurations/'  # Path for the calibration files
        data_files_path = exo_data_files+'simulations/'      	# Path for the simulation data files
else:
    detector_pickled_path = '/chroma-data/configurations/'
    detector_calibration_path = '/chroma-data/configurations/'  # Path for the calibration files
    data_files_path = '/chroma-data/simulations/'               # Path for the simulation data files


def get_calibration_file_name_without_path(config):
    return 'detresang-'+config+'_1DVariance_100million.root'

def get_calibration_file_name(config):
    return detector_calibration_path+get_calibration_file_name_without_path(config)

def get_data_file_path(config):
    return data_files_path+config+'/raw_data/'

def get_data_file_path_no_raw(config):
    return data_files_path+config+'/'
