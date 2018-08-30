import itertools,argparse,math
import scipy.spatial
import numpy as np

from lenssystem import get_system_measurements
import detectorconfig as dc
from detectorconfig import DetectorConfig  # Note: This is required for the pickle file reading, despite the fact that it is not explicitly referenced

def calc_steps(x_value,y_value,detector_r,n_lens_pixel):
        x_coord = np.asarray([x_value,np.roll(x_value,-1)]).T[:-1]
        y_coord = np.asarray([y_value,np.roll(y_value,-1)]).T[:-1]
        lat_area = 2*np.pi*detector_r*(y_coord[:,0]-y_coord[:,1])
        n_step = (lat_area/lat_area[-1]*n_lens_pixel).astype(int)
        # print('Step: ' + str(len(x_coord)) + ' ' + str(len(y_coord)) + ' ' + str(n_step))
        return x_coord, y_coord, n_step

def curved_surface2(detector_r=2.0, diameter = 2.5, nsteps=20,n_lens_pxl=4):
    '''Builds a curved surface n_lensd on the specified radius. Origin is center of surface.'''
    if (detector_r < diameter/2.0):
        raise Exception('The Radius of the curved surface must be larger than diameter/2.0')
    shift1 = np.sqrt(detector_r**2 - (diameter/2.0)**2)
    theta1 = np.arctan(shift1/(diameter/2.0))
    angles1 = np.linspace(theta1, np.pi/2, nsteps)
    x_value = abs(detector_r*np.cos(angles1))
    y_value = detector_r-detector_r*np.sin(angles1)
    # x_coord,y_coord,n_step = calc_steps(x_value,y_value,detector_r,n_lens_pixel=n_lens_pxl) # This is redundant??
    return calc_steps(x_value,y_value,detector_r,n_lens_pxl)

# This is something like: for a given number of lenses and lens radius (or max radius) and detector radius
#     Compute the total number of pixels each for from 2 to 45 rings
#     And then pick the closest one to 100,000 pixels total
def param_arr(n_lens,b_pxl,l_sys,detec_r,max_rad,target_pixel_count):
	if l_sys == 'Jiani3':
		scal_lens = 488.0/643.0
	elif l_sys == 'Sam1':
		scal_lens = 1.0
	arr,ix = [],[]
	for i in xrange(2,100):
		ix.append(i)
		arr.append(sum(curved_surface2(detec_r,2*max_rad,i,b_pxl)[2]))
	arr = np.asarray(arr)
	nstep = ix[np.argmin(np.absolute(arr*n_lens-target_pixel_count))]
	t_px = arr[np.argmin(np.absolute(arr*n_lens-target_pixel_count))]*n_lens
	#dct = np.stack((ix,scal_lens*edge_len/(2*(np.sqrt(2*px_per_face/arr).astype(int)+np.sqrt(3)-1))))
	#sel_arr = np.absolute(((n_lens*(n_lens+1))/2*arr-px_per_face)/px_per_face)<0.3
	#dct = dct[:,sel_arr]
	#if dct.shape[1]>1:
	#	dct = dct[:,np.argmin(np.absolute((n_lens*(n_lens+1))/2*arr[sel_arr]-px_per_face))]
	return nstep,t_px

def fibonacci_sphere(samples=1):
        offset = 2./samples
        increment = math.pi*(3.0-math.sqrt(5.0))
        smp = np.arange(samples)
        phi = smp * increment
        y = ((smp*offset)-1)+(offset/2)
        r = np.sqrt(1-np.square(y))
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        return np.asarray([x,y,z]).T

def acting_force(g,vtx):
        mod_arr = np.zeros((vtx.shape[0],(vtx.shape[0]-1),3))
        for i,perm in enumerate(itertools.permutations(vtx,2)):
                a = i/mod_arr.shape[1]
                b = i%mod_arr.shape[1]
                mod_arr[a,b,:] = perm[0] - perm[1]
        return g * np.einsum('ijk,ij->ik',mod_arr,1.0/np.power(np.linalg.norm(mod_arr,axis=2),3))


def calc_rad(vtx,rad_sp):
        geom_eff,i = 0,0
	g = np.exp(-2.39840805*np.log(vtx.shape[0])+5.38099281)		#obtained from a linear log-log fit see ~/Desktop/dev/dvlp/sphere.py for details
        while True:
                force = acting_force(g,vtx)
                mod_vtx = np.einsum('ij,i->ij',(vtx + force),1.0/np.linalg.norm((vtx + force),axis=1))
                rad = min(scipy.spatial.distance.pdist(mod_vtx))/2.0
                rad = rad/math.sqrt(1-math.pow(rad,2))
                new_geom_eff = (1-np.cos(np.arctan(rad)))/2.0*vtx.shape[0]
                if new_geom_eff-geom_eff<0.000005 and i>5:
                        return rad_sp*vtx, rad_sp*rd,geom_eff
                rd = rad
                vtx = mod_vtx
                geom_eff = (1-np.cos(np.arctan(rd)))/2.0*vtx.shape[0]
                i += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Calculate distributed imaging detector geometry and configuration")
    parser.add_argument('lens_system_name', help='Lens design')
    parser.add_argument('--target_pixels', '-t', default=100000, help='Target pixel count')
    args = parser.parse_args()

    lens_system_name = args.lens_system_name

    target_pixels = int(args.target_pixels)
    print('Target total pixel count: %s' % '{:,}'.format(target_pixels))
    sph_rad = 7556.0
    b_pxl = int(raw_input('input number of pixels at the central ring: (more than 3) '))
    n_lens = int(raw_input('input the number of lens assemblies: '))
    EPD_ratio = float(raw_input('input the pupil ratio: '))
    vtx,max_rad,geom_eff = calc_rad(fibonacci_sphere(n_lens),sph_rad)
    dtc_r = get_system_measurements(lens_system_name,max_rad)[1]
    ring_count,tot_pxl = param_arr(n_lens,b_pxl,lens_system_name,dtc_r,max_rad,target_pixels)

    config = dc.DetectorConfig(sph_rad, n_lens, max_rad, vtx,
                               1.0,    			# TODO: Is diameter ratio always 1.0?
                               ring_count,   	# This is actually the ring boundary count, which is one more than the number of actual rings of pixels
                               thickness_ratio=0.25,
                               blockers=True,
                               blocker_thickness_ratio=1.0/1000,
                               lens_system_name=lens_system_name,
                               EPD_ratio=EPD_ratio,
                               focal_length=1.0,
                               light_confinement=True,
                               b_pixel=b_pxl,
                               tot_pixels=tot_pxl)
    config.display_configuration()

    print '  Geometrical filling factor: %0.2f'%(geom_eff*math.pow(EPD_ratio,2))
    anw = raw_input('add %s configuration (y or n)?: '%config.config_name)
    if anw == 'y':
        cl = dc.DetectorConfigurationList()
        cl.save_configuration(config)
    elif anw == 'n':
        exit()
    '''
    print 'rings in the optical system (+1): '+str(int(a[0]))
    print 'adjusted lens radius [mm]: '+str(int(a[1]))
    print 'total amount of pixels: '+str(int(sum(curved_surface2(dtc_r,2*max_rad,a[0],b_pxl)[2])*n_lens))
    '''
