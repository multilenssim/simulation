from chroma.transform import make_rotation_matrix
import meshhelper as mh
import numpy as np
import lensmaterials as lm
from logger_lfd import logger

def thickness_of_lens_curved_part(radius_of_curvature, radius_of_lens):
    result = radius_of_curvature - pow(pow(radius_of_curvature, 2.) - pow(radius_of_lens, 2.), 0.5)
    return result

# Note: the code here uses a reference plane for constructing the lens systems and detector, which it the
# plane at the front of the cylindrical part of lens #1 (i.e. before the front curved surface).
class LensSys(object):
    def __init__ (self, name, sys_rad, detector_r_curve, lens1_rad, lens2_rad=None,
                  c1=None, c2=None, c3=None, c4=None, t1=None, t2=None,
                  gap=None, focal_surface_gap=None, epd_gap=0,
                  lensmat=lm.lensmat, focal_length=None):

        # Contains parameters common to all lens systems
        self.name = name            # Currently, this is only needed for debugging
        self._sys_rad = sys_rad # Radius of detecting surface
        #self.focal_length = focal_length # Focal length (distance from center of first lens to center of detecting surface)
        self._detector_r_curve = detector_r_curve # Radius of curvature of detecting surface
        self._lens1_rad = lens1_rad # Radius of first lens in lens system
        self._lens2_rad = lens2_rad # Radius of second lens in lens system
        self._lensmat = lensmat # Lens material

        # Radius of curvature of the lens faces from inner to outer
        self._c1 = c1
        self._c2 = c2
        self._c3 = c3
        self._c4 = c4
        # Thickness of the lenses

        logger.info('================================')
        logger.info('==== Lens system: %s ====' % self.name)

        # Temporary hack!!
        if False: # c1 is not None:  # Jiani3 uses None
            self._t1 = t1 + thickness_of_lens_curved_part(c1, lens1_rad) + thickness_of_lens_curved_part(c2, lens1_rad)
            self._t2 = t2 + thickness_of_lens_curved_part(c3, lens2_rad) + thickness_of_lens_curved_part(c4, lens1_rad)
        else:
            self._t1 = t1
            self._t2 = t2

        self._gap = gap                                  # Gap between the lenses
        self._focal_surface_gap = focal_surface_gap      # Distance from outer most lens surface to image surface - TODO: is this comment wrong??
        self._epd_gap = epd_gap                          # Distance from front of lens #1 to the iris

        if c1 is not None:  # Jiani3 uses None
            self.epd_offset = epd_gap + thickness_of_lens_curved_part(c1, lens1_rad)  # Distance form the "zero point" (edge of front curvature of lens #1) to the iris

            # TODO: Also check that the radius of curvature is not greater than diameter
            lens1_curved_thickness = thickness_of_lens_curved_part(c1, lens1_rad) + thickness_of_lens_curved_part(c2, lens1_rad)
            logger.info("Thickness of lens: %0.2f vs. thickness of curved part: %0.2f" % (t1, lens1_curved_thickness))
            if self._t1 < lens1_curved_thickness:
                logger.critical('=== Lens 1 geometry is inconsistent - overriding thickness')
                self._t1 = lens1_curved_thickness

            lens2_curved_thickness = thickness_of_lens_curved_part(c3, lens2_rad) + thickness_of_lens_curved_part(c4, lens2_rad)
            logger.info("Thickness of lens: %0.2f vs. thickness of curved part: %0.2f" % (t2, lens2_curved_thickness))
            if self._t2 < lens2_curved_thickness:
                logger.critical('=== Lens 2 geometry is inconsistent - overriding thickness')
                self._t2 = lens2_curved_thickness

        # TODO: needs testing
        # Note: this is not really the focal length.  It's just the distance from the reference plane to the detector surface along the central axis
        computed_fl = None
        if focal_length is None:
            distance_to_lens_2 = self._t1 - thickness_of_lens_curved_part(c1, lens1_rad) + gap
            self._d2 = distance_to_lens_2 + thickness_of_lens_curved_part(c3, lens2_rad)
            computed_fl = distance_to_lens_2 + self._t2 + self._focal_surface_gap
            self._focal_length = computed_fl
        else:
            try:
                # TODO: merge this duplicate code
                distance_to_lens_2 = self._t1 - thickness_of_lens_curved_part(c1, lens1_rad) + gap
                self._d2 = distance_to_lens_2 + thickness_of_lens_curved_part(c3, lens2_rad)
                computed_fl = distance_to_lens_2 + self._t2 + self._focal_surface_gap
            except Exception as e:
                if name != 'Jiani3':      # Hack to suppress Jiani3
                    logger.info('Exception computing focal length for: %s, %s' % (name, e))
                pass
            self._focal_length = focal_length
        logger.info('Focal length specified vs. computed for %s: %s, %s' % (name, str(focal_length), str(computed_fl)))
        logger.info('Image surface gap: %s' % (str(focal_surface_gap)))

    def __str__(self):
        return self.__dict__

# TODO: Do I need to retest this?
_lensdict = {'Jiani3': LensSys('Jiani3', sys_rad=643., focal_length=1074., detector_r_curve=943., c1 = None, lens1_rad=488.)}


'''
        # Old Sam1 computations
        d2 = (t1 - 111.5 * scale_factor) + (0.3 + 85.4) * scale_factor
'''

SAM1_FOCAL_SURFACE_GAP = 737.4 - 274.2 - 254.4 - 0.3 + thickness_of_lens_curved_part(605.,350.)

_lensdict['Sam1'] =      LensSys('Sam1', sys_rad=350., detector_r_curve=480., lens1_rad=350., lens2_rad=350.,
                                 c1=605.0, c2=535., c3=760., c4=782., t1=274.2, t2=254.4,
                                 gap=0.3, focal_surface_gap=SAM1_FOCAL_SURFACE_GAP,
                                 lensmat=lm.lensmat_ohara) # , focal_length=737.4)

_lensdict['Sam1-old'] =  LensSys('Sam1-old', sys_rad=350., detector_r_curve=480., lens1_rad=350., lens2_rad=350.,
                                 c1=605.0, c2=535., c3=760., c4=782., t1=274.2, t2=254.4,
                                 gap=0.3, focal_surface_gap=SAM1_FOCAL_SURFACE_GAP,
                                 lensmat=lm.lensmat_ohara, focal_length=737.4)

# TODO: use negative curvatures?
#   lens_rad is just a guess so far
#   Need to double check the "focal length" computations
_lensdict['Sam2-0.7'] =  LensSys('Sam2-0.7', sys_rad=300., detector_r_curve=600., lens1_rad=300., lens2_rad=600.,
                                 c1=932.5, c2=444.0, c3=802.2, c4=822., t1=300.03, t2=250.,
                                 gap=10.8, focal_surface_gap=384.0,
                                 lensmat=lm.lensmat_ohara)
_lensdict['Sam2-0.6'] =  LensSys('Sam2-0.6', sys_rad=300., detector_r_curve=600., lens1_rad=300., lens2_rad=600.,
                                 c1=890.7, c2=431.9, c3=797.4, c4=845.7, t1=300., t2=250.,
                                 gap=6.2, focal_surface_gap=395.7,
                                 lensmat=lm.lensmat_ohara)
_lensdict['Sam2-0.5'] =  LensSys('Sam2-0.5', sys_rad=600., detector_r_curve=600., lens1_rad=380., lens2_rad=600.,
                                 c1=744.9, c2=397.2, c3=739.1, c4=1521.5, t1=300., t2=250.,
                                 gap=22.9, focal_surface_gap=369.1, epd_gap=35.3,
                                 lensmat=lm.lensmat_ohara)
_lensdict['Sam2-0.5-600'] =  LensSys('Sam2-0.5-600', sys_rad=600., detector_r_curve=600., lens1_rad=300., lens2_rad=600.,
                                     c1=744.9, c2=397.2, c3=739.1, c4=1521.5, t1=300., t2=250.,
                                     gap=22.9, focal_surface_gap=369.1, epd_gap=35.3,
                                     lensmat=lm.lensmat_ohara)


#pprint.pprint(lensdict) # Doesn't print object fields even when __str__ is implemented

'''
for key,value in lensdict.iteritems():
    print('{\'%s\': ' % key)
    pprint.pprint(value.__dict__)
    print('}')
'''

#jsonlenses = jsonpickle.encode(lensdict)
#print(json.dumps(jsonlenses, indent=4)) # , default=lambda o: o.__dict__)      # Doesn't work because it tries to call __dict__ on everything)

def get_lens_sys(lens_system_name):
    if lens_system_name in _lensdict:
        return _lensdict[lens_system_name]
    else:
        raise Exception('Lens system name '+str(lens_system_name)+' is not valid.')

def get_scale_factor(lens_system_name, scale_rad):
    # Returns the factor used to scale a given lens system of name lens_system_name
    # Scaling will happen in such a way that all distances are set by the scale
    # factor except for the detector size, which is fixed separately.
    sys_rad = get_lens_sys(lens_system_name)._sys_rad
    sf = scale_rad/sys_rad
    logger.warning('Lens system %s scale factor: %f, %f, %f' % (lens_system_name, sf, scale_rad, sys_rad))
    return sf
  
def get_half_EPD(lens_system_name, scale_rad, EPD_ratio):
    # Returns the radius of the entrance pupil, as a fraction EPD_ratio of the first lens
    # radius, scaled up appropriately
    scale_factor = get_scale_factor(lens_system_name, scale_rad)
    lens_sys = get_lens_sys(lens_system_name)
    lens1_rad_unscaled = lens_sys._lens1_rad
    lens1_rad = scale_factor*lens1_rad_unscaled
    return lens1_rad*EPD_ratio
  
def get_system_measurements(lens_system_name, scale_rad):
    # Returns the focal length and detecting surface radius of curvature for 
    # a given lens system of name lens_system_name; 
    scale_factor = get_scale_factor(lens_system_name, scale_rad)
    lens_sys = get_lens_sys(lens_system_name)
    fl_unscaled = lens_sys._focal_length
    det_r_unscaled = lens_sys._detector_r_curve
    fl = fl_unscaled*scale_factor
    det_r = det_r_unscaled*scale_factor
    
    return fl, det_r
    #det_diam < 2*det_r = 2*sys_det_r*scale_factor

def get_lens_mesh_list(lens_system_name, scale_rad):
    # Returns a list of lens meshes for the given lens_system_name, scaled to match scale_rad
    lenses = []
    lens_sys = get_lens_sys(lens_system_name)
    scale_factor = get_scale_factor(lens_system_name, scale_rad)
    logger.warning('Scale factor for lens %s: %f' % (lens_system_name, scale_factor))
    if lens_system_name == 'Jiani3':
        # Asphere - Jiani design 3
        as_rad = lens_sys.lens_rad*scale_factor
        #print "asphere radius: ", as_rad
        diameter = 2.0*as_rad
        as_t = 506.981*scale_factor
        
        as_c1 = (1./820.77)/scale_factor
        as_k1 = -7.108
        as_d1 = 2.028e-4/scale_factor
        as_e1 = -1.294e-9/(scale_factor**3)
        as_f1 = 1.152e-15/(scale_factor**5)
        
        as_c2 = (-1./487.388)/scale_factor
        as_k2 = -0.078
        as_d2 = -2.412e-4/scale_factor
        as_e2 = 9.869e-10/(scale_factor**3)
        as_f2 = -1.49e-15/(scale_factor**5)
        
        as_mesh = mh.rotate(asphere_lens(as_rad, as_t, as_c1, as_k1, as_d1, as_e1, as_f1, as_c2, as_k2, as_d2, as_e2, as_f2, nsteps=64), make_rotation_matrix(np.pi/2, (1,0,0)))
        lenses.append(as_mesh)

    elif lens_system_name == 'Sam1-old':

        # First lens
        lens1rad = lens_sys._lens1_rad * scale_factor
        t1 = 274.2 * scale_factor
        lens1_c1 = (1. / 605.) / scale_factor
        lens1_c2 = (-1. / 535.) / scale_factor
        l1_mesh = mh.rotate(asphere_lens(lens1rad, t1, lens1_c1, 0., 0., 0., 0., lens1_c2, 0., 0., 0., 0., nsteps=64),
                            make_rotation_matrix(np.pi / 2, (1, 0, 0)))
        lenses.append(l1_mesh)

        # Second lens
        lens2rad = lens_sys._sys_rad * scale_factor
        t2 = 254.4 * scale_factor
        lens2_c1 = (1. / 760.) / scale_factor
        lens2_c2 = (-1. / 782.) / scale_factor
        l2_mesh = mh.rotate(asphere_lens(lens2rad, t2, lens2_c1, 0., 0., 0., 0., lens2_c2, 0., 0., 0., 0., nsteps=64),
                            make_rotation_matrix(np.pi / 2, (1, 0, 0)))

        d2_unscaled = lens_sys._t1 - thickness_of_lens_curved_part(lens_sys._c1, lens_sys._lens1_rad) + lens_sys._gap + thickness_of_lens_curved_part(lens_sys._c3, lens_sys._lens2_rad)
        logger.info('Unscaled and scaled d2: %f, %f' % (d2_unscaled, d2_unscaled*scale_factor))

        distance_to_lens_2 = lens_sys._t1 - thickness_of_lens_curved_part(lens_sys._c1, lens_sys._lens1_rad) + lens_sys._gap
        focal_surface_gap = 737.4 - distance_to_lens_2 - lens_sys._t2
        logger.info('Unscaled and scaled focal surface gap: %f, %f' % (focal_surface_gap, focal_surface_gap*scale_factor))
        
        d2_orig = (t1 - 111.5 * scale_factor) + (0.3 + 85.4) * scale_factor
        d2 = (t1 - thickness_of_lens_curved_part(lens_sys._c1, lens_sys._lens1_rad) * scale_factor) + (lens_sys._gap + thickness_of_lens_curved_part(lens_sys._c3, lens_sys._lens2_rad)) * scale_factor
        logger.info('Orig and alternative d2: %f, %f' % (d2_orig, d2))   # These differ just because of rounding in the 111.5 and 85.4
        logger.info('Lens curved part thicknesses: %f, %f' % (thickness_of_lens_curved_part(lens_sys._c1, lens_sys._lens1_rad), thickness_of_lens_curved_part(lens_sys._c3, lens_sys._lens2_rad)))
        l2_mesh = mh.shift(l2_mesh, (0., 0., -d2))  # Shift relative to first lens along optical axis
        lenses.append(l2_mesh)

    else:
        lens1rad = lens_sys._lens1_rad * scale_factor
        # First lens
        t1 = lens_sys._t1 * scale_factor
        lens1_c1 = (1. / lens_sys._c1) / scale_factor
        lens1_c2 = (-1. / lens_sys._c2) / scale_factor

        l1_mesh = mh.rotate(asphere_lens(lens1rad, t1, lens1_c1, 0., 0., 0., 0., lens1_c2, 0., 0., 0., 0., nsteps=64),
                            make_rotation_matrix(np.pi / 2, (1, 0, 0)))
        lenses.append(l1_mesh)

        # Second lens
        lens2rad = lens_sys._lens2_rad * scale_factor

        t2 = lens_sys._t2 * scale_factor
        lens2_c1 = (1. / lens_sys._c3) / scale_factor
        lens2_c2 = (-1. / lens_sys._c4) / scale_factor

        l2_mesh = mh.rotate(asphere_lens(lens2rad, t2, lens2_c1, 0., 0., 0., 0., lens2_c2, 0., 0., 0., 0., nsteps=64),
                            make_rotation_matrix(np.pi / 2, (1, 0, 0)))
        #d2 = (t1 - thickness_of_lens_curved_part(lens_sys.c1, lens_sys._lens1_rad) * scale_factor) + (lens_sys.gap + thickness_of_lens_curved_part(lens_sys.c3, lens_sys._lens2_rad)) * scale_factor
        d2_scaled = lens_sys._d2 * scale_factor
        # print('=== d2 for %s: %f, unscaled: %f, scaled precomputed: %f' % (lens_system_name,d2, lens_sys.d2, d2_pre))
        l2_mesh = mh.shift(l2_mesh, (0., 0., -d2_scaled))  # Shift relative to first lens along optical axis
        lenses.append(l2_mesh)

    return lenses
        
def get_lens_material(lens_system_name):
    # Returns the lens material for the given lens_system_name
    lens_sys = get_lens_sys(lens_system_name)
    return lens_sys._lensmat
        
def asphere_func(x, c, k, d, e, f):
    # General aspheric lens parametrization
    return c*(x**2)/(1+np.sqrt(1-(1+k)*(c**2)*(x**2))) + d*(x**2) + e*(x**4) + f*(x**6)

def asphere_lens(rad, t, c1, k1, d1, e1, f1, c2, k2, d2, e2, f2, nsteps=128):
    # Returns a mesh of an aspheric lens with two surfaces parametrized by c, k, d, e, f
    # rad - radius of lens
    # t - thickness at widest point (if this is too small for the desired rad, will use min consistent thickness)
    # c - 1/(radius of curvature) in 1/mm
    # k - conic constant (unitless)
    # d - quadratic (1/mm); e - quartic (1/mm^3); f - sextic (1/mm^5)
    # First surface should be before second, so c1 > c2 must hold (for all signs of c1 and c2)
    # Note that parameters do not all scale by the same factor (not linear)
    # To multiply lens size by a, multiply diameter by a, divide c and d by a, divide e by a^3, etc.
    # Center at X=0, Y=0; rotated about x-axis, Y=0 plane is edge of first surface
    ymax_1 = asphere_func(rad, c1, k1, d1, e1, f1)
    ymax_2 = asphere_func(rad, c2, k2, d2, e2, f2)
    if t < ymax_1 - ymax_2: # Avoid situations where the two lens faces overlap each other
        #print ymax_1, ymax_2, ymax_1-ymax_2, t
        t = ymax_1 - ymax_2
 
    x1 = np.linspace(0., rad, nsteps/2) # Space out points linearly in x
    x2 = np.linspace(rad, 0., nsteps/2)
    
    y1 = asphere_func(x1, c1, k1, d1, e1, f1)-ymax_1 # Shift so Y=0 plane is the first surface's edge
    y2 = asphere_func(x2, c2, k2, d2, e2, f2)+t-ymax_1 # Shift second surface to give appropriate thickness

    from chroma import make
    return make.rotate_extrude(np.concatenate((x1, x2)), np.concatenate((y1, y2)), nsteps=64)
