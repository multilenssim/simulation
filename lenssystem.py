from chroma import make
from chroma.transform import make_rotation_matrix
import meshhelper as mh
import numpy as np

class LensSys(object):
    def __init__ (self, sys_rad, focal_length, detector_r_curve, lens_rad):
        # Contains parameters common to all lens systems
        self.sys_rad = sys_rad # Radius of detecting surface
        self.focal_length = focal_length # Focal length (distance from center of first lens to center of detecting surface)
        self.detector_r_curve = detector_r_curve # Radius of curvature of detecting surface
        self.lens_rad = lens_rad # Radius of first lens in lens system
        
lensdict = {'Jiani3': LensSys(sys_rad=643., focal_length=1074., detector_r_curve=943., lens_rad=488.)}
lensdict['Sam1'] = LensSys(sys_rad=350., focal_length=737.4, detector_r_curve=480., lens_rad=350.)

def get_lens_sys(lens_system_name):
    if lens_system_name in lensdict:
        return lensdict[lens_system_name]
    else:
        raise Exception('Lens system name '+str(lens_system_name)+' is not valid.')

def get_scale_factor(lens_system_name, scale_rad):
	# Returns the factor used to scale a given lens system of name lens_system_name
    # Scaling will happen in such a way that all distances are set by the scale
    # factor except for the detector size, which is fixed separately.
    sys_rad = get_lens_sys(lens_system_name).sys_rad
    return scale_rad/sys_rad
  
def get_half_EPD(lens_system_name, scale_rad, EPD_ratio):
    # Returns the radius of the entrance pupil, as a fraction EPD_ratio of the first lens
    # radius, scaled up appropriately
    scale_factor = get_scale_factor(lens_system_name, scale_rad)
    lens_sys = get_lens_sys(lens_system_name)
    lens_rad_unscaled = lens_sys.lens_rad
    lens_rad = scale_factor*lens_rad_unscaled
    return lens_rad*EPD_ratio
  
def get_system_measurements(lens_system_name, scale_rad):
    # Returns the focal length and detecting surface radius of curvature for 
    # a given lens system of name lens_system_name; 
    scale_factor = get_scale_factor(lens_system_name, scale_rad)
    lens_sys = get_lens_sys(lens_system_name)
    fl_unscaled = lens_sys.focal_length
    det_r_unscaled = lens_sys.detector_r_curve
    fl = fl_unscaled*scale_factor
    det_r = det_r_unscaled*scale_factor
    
    return fl, det_r
    #det_diam < 2*det_r = 2*sys_det_r*scale_factor

def get_lens_mesh_list(lens_system_name, scale_rad):
    # Returns a list of lens meshes for the given lens_system_name, scaled to match scale_rad
    lenses = []
    lens_sys = get_lens_sys(lens_system_name)
    scale_factor = get_scale_factor(lens_system_name, scale_rad)
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
    elif lens_system_name == 'Sam1':
        
        lens1rad = lens_sys.lens_rad*scale_factor
        # First lens
        t1 = 274.2*scale_factor
        lens1_c1 = (1./605.)/scale_factor
        lens1_c2 = (-1./535.)/scale_factor
        
        l1_mesh = mh.rotate(asphere_lens(lens1rad, t1, lens1_c1, 0., 0., 0., 0., lens1_c2, 0., 0., 0., 0., nsteps=64), make_rotation_matrix(np.pi/2, (1,0,0)))
        lenses.append(l1_mesh)
        
        # Second lens
        lens2rad = lens_sys.sys_rad*scale_factor
        
        t2 = 254.4*scale_factor
        lens2_c1 = (1./760.)/scale_factor
        lens2_c2 = (-1./782.)/scale_factor
        
        l2_mesh = mh.rotate(asphere_lens(lens2rad, t2, lens2_c1, 0., 0., 0., 0., lens2_c2, 0., 0., 0., 0., nsteps=64), make_rotation_matrix(np.pi/2, (1,0,0)))
        d2 = (t1-111.5*scale_factor)+(0.3+85.4)*scale_factor
        l2_mesh = mh.shift(l2_mesh, (0., 0., -d2)) # Shift relative to first lens along optical axis
        lenses.append(l2_mesh)
    return lenses
        
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
    
    return make.rotate_extrude(np.concatenate((x1, x2)), np.concatenate((y1, y2)), nsteps=64)
