from lenssystem import get_system_measurements, get_half_EPD
from paths import detector_pickled_path
import pickle

import driver_utils

class DetectorConfig(object):
    def __init__ (self, sph_rad, n_lens, max_radius, vtx,
                  pmtxbins, pmtybins,
                  diameter_ratio,
                  thickness_ratio=0.25,
                  blockers=True,
                  blocker_thickness_ratio=1.0/1000,
                  lens_system_name=None,
                  EPD_ratio=1.0,
                  focal_length=1.0,
                  light_confinement=False,
                  nsteps=10,
                  b_pixel=4,
                  tot_pixels=None,
                  uuid=None,
                  config_name=None):
        # Spherical geometry the focal surface is considered curve (planar surface not considered).
        # The parameters of the lens system defined in lenssystem.py for any given name. 
        # Note that focal_length can only be explicitly set for planar detectors (otherwise lenssystem
        # parameters override it).
        # The attributes' names are the same of the icosahedron to have the scripts compatible
        self.EPD_ratio = EPD_ratio
        self.edge_length = sph_rad # Radius of the detector
        self.base = n_lens # Total number of lenses
        self.pmtxbins = pmtxbins # If planar detector, bins in x for each plane; if curved, does nothing
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
        self.nsteps = nsteps # number of steps to generate curved detecting surface - sets number of PMTs
        self.b_pixel = b_pixel # number of pixels in the first ring (active only with NEW PIXELIZATION)
        self.vtx = vtx
        self.tot_pixels = tot_pixels
        self.uuid = uuid
        self.config_name = config_name


# All configpc diameters given are for kabamlandpc packing (different for kabamland2) 
# Up to configpc4, focal_length=diameter
# Configs up to configpc4 should be considered deprecated
# The other configpc cases do not set focal_length or light_confinement here due to historical reasons;
# filenames using them should specify both of these parameters; future runs should just create new configs

def get_dict_param(conf_fl,conf_name):
    with open(conf_fl,'r') as f:
        dtc = pickle.load(f)
    return dtc[conf_name]
'''
# Newer configurations, including those with curved detecting surfaces and pre-made lens systems
configdict = {'cfJiani3_2': DetectorConfig(10000.0, 6, 0, 0, 1.0, lens_system_name='Jiani3', light_confinement=True, nsteps=23)} # Should have ~100k pixels, 21 lens systems/face, 20 faces
configdict['cfJiani3_4'] = DetectorConfig(10000.0, 4, 0, 0, 1.0, lens_system_name='Jiani3', light_confinement=True, nsteps=32) # Should have ~100k pixels, 10 lens systems/face, 20 faces
configdict['cfJiani3_8'] = DetectorConfig(10000.0, 4, 0, 0, 1.0, lens_system_name='Jiani3', light_confinement=True, nsteps=11, b_pixel=5) #94600 pxl,10 system/face,l_radius: 80cm NP
configdict['cfJiani3_9'] = DetectorConfig(10000.0, 3, 0, 0, 1.0, lens_system_name='Jiani3', light_confinement=True, nsteps=16, b_pixel=4) #102120 pxl,6 system/face,l_radius: 102cm NP
configdict['cfSam1_1'] = DetectorConfig(10000.0, 6, 0, 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=23) # Should have ~100k pixels, 21 lens systems/face, 20 faces
configdict['cfSam1_2'] = DetectorConfig(10000.0, 4, 0, 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=23) # Should have ~100k pixels, 10 lens systems/face, 20 faces
configdict['cfSam1_3'] = DetectorConfig(10000.0, 9, 0, 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=12) # Should have ~100k pixels, 45 lens systems/face, 20 faces
configdict['cfSam1_4'] = DetectorConfig(10000.0, 10, 0, 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=12) # Should have ~100k pixels, 55 lens systems/face, 20 faces
configdict['cfSam1_5'] = DetectorConfig(10000.0, 4, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=11, b_pixel=5) #93800plx, 10 system/face, l_radius: 105cm np

configdict['cfSam1_K4_10'] = DetectorConfig(10000.0, 4, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=13, b_pixel=4) #107600plx, 10 system/face, l_radius: 105cm np
configdict['cfSam1_K2_10'] = DetectorConfig(10000.0, 2, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=22, b_pixel=4) #99420plx, 3 system/face, l_radius: 183cm np
configdict['cfSam1_K6_10'] = DetectorConfig(10000.0, 6, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=9, b_pixel=4) #99960plx, 21 system/face, l_radius: 74cm np
configdict['cfSam1_K1_10'] = DetectorConfig(10000.0, 1, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=37, b_pixel=4) #97620plx, 1 system/face, l_radius: 289cm np
configdict['cfSam1_K8_10'] = DetectorConfig(10000.0, 8, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=7, b_pixel=4) #95760plx, 1 system/face, l_radius: 57cm np
configdict['cfSam1_K10_10'] = DetectorConfig(10000.0, 10, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=6, b_pixel=4) #101200plx, 55 system/face, l_radius: 46cm np
configdict['cfSam1_K10_8'] = DetectorConfig(10000.0, 10, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=6, b_pixel=4) #101200plx, 55 system/face, l_radius: 46cm np
configdict['cfSam1_K2_8'] = DetectorConfig(10000.0, 2, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=22, b_pixel=4)
configdict['cfSam1_K6_8'] = DetectorConfig(10000.0, 6, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=9, b_pixel=4)
configdict['cfSam1_K8_8'] = DetectorConfig(10000.0, 8, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=7, b_pixel=4)
configdict['cfSam1_K4_8'] = DetectorConfig(10000.0, 4, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=13, b_pixel=4)
configdict['cfSam1_K12_10'] = DetectorConfig(10000.0, 12, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=5, b_pixel=4) #92040plx, 78 system/face, l_radius: 36cm np
configdict['cfSam1_K12_8'] = DetectorConfig(10000.0, 12, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=5, b_pixel=4) #92040plx, 78 system/face, l_radius: 36cm np
configdict['cfSam1_K4_5'] = DetectorConfig(10000.0, 4, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.5, light_confinement=True, nsteps=13, b_pixel=4)
configdict['cfSam1_K1_8'] = DetectorConfig(10000.0, 1, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=37, b_pixel=4)

configdict['cfSam1_k1_10'] = DetectorConfig(10000.0, 1, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=13, b_pixel=4) #10760plx, 1 system/face, l_radius: 289cm np
configdict['cfSam1_k2_10'] = DetectorConfig(10000.0, 2, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=8, b_pixel=4) #10980plx, 3 system/face, l_radius: 183cm np
configdict['cfSam1_k3_10'] = DetectorConfig(10000.0, 3, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=6, b_pixel=4) #11040plx, 6 system/face, l_radius: 134cm np
configdict['cfSam1_k4_10'] = DetectorConfig(10000.0, 4, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=5, b_pixel=4) #11800plx, 10 system/face, l_radius: 106cm np
configdict['cfSam1_k6_10'] = DetectorConfig(10000.0, 6, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=3, b_pixel=6) #9660plx, 21 system/face, l_radius: 74cm np
configdict['cfSam1_k1_8'] = DetectorConfig(10000.0, 1, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=13, b_pixel=4) #10760plx, 1 system/face, l_radius: 289cm np
configdict['cfSam1_k2_8'] = DetectorConfig(10000.0, 2, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=8, b_pixel=4) #10980plx, 3 system/face, l_radius: 183cm np
configdict['cfSam1_k3_8'] = DetectorConfig(10000.0, 3, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=6, b_pixel=4) #11040plx, 6 system/face, l_radius: 134cm np
configdict['cfSam1_k4_8'] = DetectorConfig(10000.0, 4, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=5, b_pixel=4) #11800plx, 10 system/face, l_radius: 106cm np
configdict['cfSam1_k6_8'] = DetectorConfig(10000.0, 6, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=3, b_pixel=6) #9660plx, 21 system/face, l_radius: 74cm np

configdict['cfSam1_M4_10'] = DetectorConfig(10000.0, 4, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=37, b_pixel=4)
configdict['cfSam1_M20_10'] = DetectorConfig(10000.0, 20, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=9, b_pixel=4)
configdict['cfSam1_M20_8'] = DetectorConfig(10000.0, 20, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=9, b_pixel=4)
configdict['cfSam1_M4_8'] = DetectorConfig(10000.0, 4, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=37, b_pixel=4) #976200plx, 10 system/face, l_radius: 1056cm np
configdict['cfSam1_M10_10'] = DetectorConfig(10000.0, 10, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=17, b_pixel=4) #1056000 plx, 55 system/face, l_radius: 465 cm np
configdict['cfSam1_M10_8'] = DetectorConfig(10000.0, 10, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=17, b_pixel=4) #1056000 plx, 55 system/face, l_radius: 465 cm np
'''
''' Old configuration names - maintained here for reference so that we can rename the config files
configdict['cfSam1_6'] = DetectorConfig(10000.0, 4, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=13, b_pixel=4) #107600plx, 10 system/face, l_radius: 105cm np
configdict['cfSam1_7'] = DetectorConfig(10000.0, 2, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=22, b_pixel=4) #99420plx, 3 system/face, l_radius: 183cm np
configdict['cfSam1_8'] = DetectorConfig(10000.0, 6, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=9, b_pixel=4) #99960plx, 21 system/face, l_radius: 74cm np
configdict['cfSam1_9'] = DetectorConfig(10000.0, 1, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=37, b_pixel=4) #97620plx, 1 system/face, l_radius: 289cm np
configdict['cfSam1_10'] = DetectorConfig(10000.0, 8, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=7, b_pixel=4) #95760plx, 1 system/face, l_radius: 57cm np
configdict['cfSam1_11'] = DetectorConfig(10000.0, 10, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=6, b_pixel=4) #101200plx, 55 system/face, l_radius: 46cm np
configdict['cfSam1_12'] = DetectorConfig(10000.0, 2, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=22, b_pixel=4) #99420plx, 3 system/face, l_radius: 183cm np
configdict['cfSam1_13'] = DetectorConfig(10000.0, 6, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=9, b_pixel=4) #99960plx, 21 system/face, l_radius: 74cm np
configdict['cfSam1_14'] = DetectorConfig(10000.0, 8, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=7, b_pixel=4) #95760plx, 1 system/face, l_radius: 57cm np
configdict['cfSam1_15'] = DetectorConfig(10000.0, 10, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=6, b_pixel=4) #101200plx, 55 system/face, l_radius: 46cm np
configdict['cfSam1_16'] = DetectorConfig(10000.0, 12, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=5, b_pixel=4) #92040plx, 78 system/face, l_radius: 36cm np
configdict['cfSam1_17'] = DetectorConfig(10000.0, 12, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=5, b_pixel=4) #92040plx, 78 system/face, l_radius: 36cm np
configdict['cfSam1_18'] = DetectorConfig(10000.0, 10, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=6, b_pixel=4) #101200plx, 55 system/face, l_radius: 46cm np
configdict['cfSam1_21'] = DetectorConfig(10000.0, 20, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=9, b_pixel=4) #~1 Million, 20 lens at base
# Note that Sam1_22 corresponds to Sam1_19
configdict['cfSam1_22'] = DetectorConfig(10000.0, 4, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=37, b_pixel=4) #976200plx, 10 system/face, l_radius: 1056cm np
''''''
configdict['cfSam1_19'] = DetectorConfig(10000.0, 4, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=37, b_pixel=4) #976200plx, 10 system/face, l_radius: 1056cm np
configdict['cfSam1_20'] = DetectorConfig(10000.0, 20, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=9, b_pixel=4) #~1 Million, 20 lens at base
configdict['cfSam1_23'] = DetectorConfig(10000.0, 10, 0 , 0, 1.0, lens_system_name='Sam1', light_confinement=True, nsteps=17, b_pixel=4) #1056000 plx, 55 system/face, l_radius: 465 cm np
''''''
configdict['cfSam1_24'] = DetectorConfig(10000.0, 10, 0 , 0, 1.0, lens_system_name='Sam1', EPD_ratio = 0.8, light_confinement=True, nsteps=17, b_pixel=4) #1056000 plx, 55 system/face, l_radius: 465 cm np
'''

def configdict(conf_name):
    fname  =  '%sconf_file.p'%detector_pickled_path
    config_dict = get_dict_param(fname,conf_name)
    return driver_utils.detector_config_from_parameter_array(conf_name, config_dict, lens_system_name='Sam1', light_confinement=True)
