from lenssystem import get_system_measurements, get_half_EPD
import numpy as np

class DetectorConfig(object):
    def __init__ (self, edge_length, base, pmtxbins, pmtybins, diameter_ratio, thickness_ratio=0.25, blockers=True, blocker_thickness_ratio=1.0/1000, lens_system_name=None, EPD_ratio=1.0, focal_length=1.0, light_confinement=False, nsteps=10,b_pixel=4):
        # Two configurations are considered here: one with a single planar detecting surface for each
        # icosahedral side (detector_r=0.) and one with curved detecting surfaces behind each lens system,
        # with the parameters of the lens system defined in lenssystem.py for any given name. 
        # Note that focal_length can only be explicitly set for planar detectors (otherwise lenssystem
        # parameters override it).
	self.EPD_ratio = EPD_ratio
        self.edge_length = edge_length # Length of an icosahedral edge for the lens plane
        self.base = base # Number of lens systems at the base of each lens plane triangle
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
        #max_radius = find_max_radius(edge_length, base)
        max_radius = edge_length/(2*(base+np.sqrt(3)-1))
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

# All configpc diameters given are for kabamlandpc packing (different for kabamland2) 
# Up to configpc4, focal_length=diameter
# Configs up to configpc4 should be considered deprecated
# The other configpc cases do not set focal_length or light_confinement here due to historical reasons;
# filenames using them should specify both of these parameters; future runs should just create new configs


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
