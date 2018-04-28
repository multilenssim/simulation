import pickle
import argparse
import pprint
import uuid

from logger_lfd import logger
from lenssystem import get_system_measurements, get_half_EPD
import paths

class DetectorConfig(object):
    def __init__ (self, sph_rad, n_lens, max_radius, vtx,
                  diameter_ratio,
                  ring_count,
                  pmtxbins=None, pmtybins=None,   # Only used for flat detector surface
                  thickness_ratio=0.25,
                  blockers=True,
                  blocker_thickness_ratio=1.0/1000,
                  lens_system_name=None,
                  EPD_ratio=1.0,
                  focal_length=1.0,
                  light_confinement=False,
                  b_pixel=4,
                  tot_pixels=None,
                  the_uuid=None,
                  config_name=None):
        # Spherical geometry the focal surface is considered curve (planar surface not considered).
        # The parameters of the lens system defined in lenssystem.py for any given name. 
        # Note that focal_length can only be explicitly set for planar detectors (otherwise lenssystem
        # parameters override it).
        # The attributes' names are the same of the icosahedron to have the scripts compatible
        self.EPD_ratio = EPD_ratio
        self.detector_radius = sph_rad  # Radius of the detector
        self.lens_count = n_lens        # Total number of lenses
        self.pmtxbins = pmtxbins        # If planar detector, bins in x for each plane; if curved, does nothing
        self.pmtybins = pmtybins
        # For planar detector, diameter_ratio is the ratio of lens system diameter that for max packing 
        # For curved detecting surfaces, all lens system distances will be scaled by diameter_ratio 
        # aside from the detector surface size (but including detector radius of curvature) 
        # Hence, diameter_ratio=1.0 means use the ratio set by the default lens system parameters.
        self.diameter_ratio = diameter_ratio         
        self.thickness_ratio = thickness_ratio # Sets lens params for old configs; deprecated
        self.blockers = blockers # Toggles blocking surfaces at the lens plane between lenses
        self.blocker_thickness_ratio = blocker_thickness_ratio # Sets thickness of blockers; should have no effect
        self.lens_system_name = lens_system_name
        self.max_radius = max_radius
        if lens_system_name:
            # Get focal length and radius of curvature of detecting surfaces; detecting surface will
            # extend to the maximum allowable radius, but its radius of curvature (and all other lens
            # system parameters) will scale with the diameter_ratio
            self.focal_length, self.detector_r = get_system_measurements(lens_system_name, max_radius*diameter_ratio)
            self.half_EPD = get_half_EPD(lens_system_name, max_radius*diameter_ratio, self.EPD_ratio)
        else: # Default, for flat detecting surface
            self.focal_length = focal_length # Use given focal length
            self.detector_r = 0. # Flat detector
            self.half_EPD = diameter_ratio*max_radius # Just uses diameter_ratio instead of EPD_ratio, for backwards compatibility
        self.light_confinement = light_confinement
        self.ring_count = ring_count # number of steps to generate curved detecting surface - sets number of PMTs - I think this is now the number of rings
        self.base_pixels = b_pixel # number of pixels in the first ring (active only with NEW PIXELIZATION)
        self.vtx = vtx
        self.tot_pixels = tot_pixels
        if the_uuid is None:
            self.uuid = uuid.uuid1()
        else:
            self.uuid = the_uuid
        if config_name is None:
            # Note: lens_system_name could be None
            self.config_name = 'cf%s_l%i_p%i_b%i_e%i' % (self.lens_system_name, self.lens_count, self.tot_pixels, self.base_pixels, int(self.EPD_ratio*10))
        else:
            self.config_name = config_name

    def display_configuration(self):
        print('=== Config: %s =====' % self.config_name)
        print ('  Detector radius:\t%0.2f'  % self.detector_radius)
        print ('  Number of lenses:\t%d'    % self.lens_count)
        print ('  Max radius:\t\t%0.2f'     % self.max_radius)
        print ('  EPD ratio:\t\t%0.2f'      % self.EPD_ratio)
        print ('  Number of rings (+1):\t%d' % self.ring_count)  # TODO: Is this really +1
        print ('  Central pixels:\t%d'      % self.base_pixels)
        print ('  UUID:\t\t\t\t%s'          % str(self.uuid))
        print ('  Total pixels in detector:\t%s'  % '{:,}'.format(self.tot_pixels))

        #lens_system_name = config_name.split('_')[0][2:]
        #dtc_r = get_system_measurements(lens_system_name, max_rad)[1]
        #n_step, tot_pxl = param_arr(n_lens, b_pxl, lens_system_name, dtc_r, max_rad)
        #print ('  Total pixels (computed):\t%d'       % tot_pxl)

        print('-------------------------')

_config_list = None
_configs_pickle_file = '%sconf_file_obj.pickle' % paths.detector_config_path

# Maintains the detector configuration file
class DetectorConfigurationList(object):
    def __init__(self):
        global _config_list, _config_pickle_file

        if _config_list is None:
            try:
                with open(_configs_pickle_file,'r') as f:
                    _config_list = pickle.load(f)
                logger.info('Loaded config list: %s' % _configs_pickle_file)
            except IOError:
                _config_list = {}
                self._save_config_list()

    def _save_config_list(self):
        try:
            with open(_configs_pickle_file,'w') as f:
                pickle.dump(_config_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        except IOError:  # Needs to be tested
            logger.critical('Unable to read or write configuration list: %s.  Quitting.' % _configs_pickle_file)
            exit(-1)

    def save_configuration(self, config):
        global _config_list

        conf_name = config.config_name
        if conf_name in _config_list:
            logger.info('Replacing configuration: ' + conf_name)
        _config_list[conf_name] = config
        self._save_config_list()
        logger.info('Configuration saved: ' + conf_name)

    def get_configuration(self, name):
        return _config_list[name]

    def _get_dict(self):  # Only intended for use by the display routine below!
        return _config_list

# Convenience method so don't have to create a list (singleton) each time
def get_detector_config(config_name):
    if _config_list is None:
        cl = DetectorConfigurationList()
        return cl.get_configuration(config_name)
    else: # this inconsistency shold be improved XXXX
        return _config_list[config_name]

# All configpc diameters given are for kabamlandpc packing (different for kabamland2) 
# Up to configpc4, focal_length=diameter
# Configs up to configpc4 should be considered deprecated
# The other configpc cases do not set focal_length or light_confinement here due to historical reasons;
# filenames using them should specify both of these parameters; future runs should just create new configs

# The old parameters:    edge_length, base, pmtxbins, pmtybins, diameter_ratio,

#
# Display lists of available detector configurations
#
if __name__ == '__main__':
    global _configs_pickle_file

    parser = argparse.ArgumentParser('Display detector configurations')
    parser.add_argument('--simple_list', '-s', action='store_true', help='List only the detector configuration names in the configuration file')
    parser.add_argument('--full_list', '-f', action='store_true', help='List the full detector configurations')
    parser.add_argument('--convert', '-c', action='store_true', help='Convert from old style array based configuration, to new styleobject-based')
    args = parser.parse_args()

    if args.convert:
        _configs_pickle_file = '%sconf_file.p' % paths.detector_config_path
        new_config_dict = {}
        with open(_configs_pickle_file, 'r') as f:
            dct = pickle.load(f)
            for key, value in dct.iteritems():
                det_config = DetectorConfig(value[0],
                                    value[1],
                                    value[2],
                                    value[3],
                                    1.0,
                                    value[5],
                                    lens_system_name='Sam1',
                                    EPD_ratio=value[4],
                                    light_confinement=True,
                                    b_pixel=value[6],
                                    tot_pixels=value[7] if len(value) > 7 else None,
                                    the_uuid=value[8] if len(value) > 8 else None,
                                    config_name=key)
                new_config_dict[key] = det_config
        with open(_configs_pickle_file+'.2.pickle','w') as f:
            pickle.dump(new_config_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        cl = DetectorConfigurationList()
        dct = cl._get_dict()
        for key, value in dct.iteritems():
            if args.simple_list:
                print(key)
            else:
                value.display_configuration()
                if args.full_list:
                    pprint.pprint(vars(value))
                    print('========================')
