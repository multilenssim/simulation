# Note that the code assumes that there is a trailing '/' on these
detector_pickled_path = '/chroma-data/configurations/'
detector_calibration_path = '/chroma-data/configurations/' # This is the path for the calibration files
data_files_path = '/chroma-data/simulations/'

def get_calibration_file_name(config):
    return detector_calibration_path+'detresang-'+config+'_1DVariance_100million.root'

# This is a hack - need to clean up
def get_calibration_file_name_without_path(config):
    return 'detresang-'+config+'_1DVariance_100million.root'


def get_data_file_path(config):
    return data_files_path+config+'/raw_data/'

    
