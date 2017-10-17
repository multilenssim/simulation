pickled_detector_path = './pickled_detectors/'
detector_configuration_path = '../TestData/'
data_files_path = '../TestData/'

def get_config_file_name(config):
    return detector_configuration_path+'detresang-'+config+'_1DVariance_100million.root'

def get_data_file_path(config):
    return data_files_path+config+'/raw_data/'

    
