from lenssystem import get_system_measurements, get_half_EPD
from paths import detector_pickled_path
import numpy as np
import pickle

class DetectorConfig(object):
    def __init__ (self, sph_rad, n_lens, max_radius, vtx, pmtxbins, pmtybins, diameter_ratio, thickness_ratio=0.25, blockers=True, blocker_thickness_ratio=1.0/1000, lens_system_name=None, EPD_ratio=1.0, focal_length=1.0, light_confinement=False, nsteps=10,b_pixel=4):
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

# All configpc diameters given are for kabamlandpc packing (different for kabamland2) 
# Up to configpc4, focal_length=diameter
# Configs up to configpc4 should be considered deprecated
# The other configpc cases do not set focal_length or light_confinement here due to historical reasons;
# filenames using them should specify both of these parameters; future runs should just create new configs

def get_dict_param(conf_fl,conf_name):
	with open(conf_fl,'r') as f:
		dtc = pickle.load(f)
	return dtc[conf_name]
# Newer configurations, including those with curved detecting surfaces and pre-made lens systems
def configdict(conf_name):
	fname  =  '%sconf_file.p'%detector_pickled_path
	return DetectorConfig(get_dict_param(fname,conf_name)[0], get_dict_param(fname,conf_name)[1], get_dict_param(fname,conf_name)[2], get_dict_param(fname,conf_name)[3],0, 0, 1.0, lens_system_name='Sam1', EPD_ratio = get_dict_param(fname,conf_name)[4], light_confinement=True, nsteps=get_dict_param(fname,conf_name)[5],b_pixel=get_dict_param(fname,conf_name)[6])
