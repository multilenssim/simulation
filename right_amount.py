import itertools,argparse,math,pickle
import scipy.spatial
import numpy as np
import uuid

from lenssystem import get_system_measurements
from paths import detector_pickled_path

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

# Something like: for a given number of lenses and lens radius (or max radius) and detector radius
#   Compute the total number of pixels each for from 2 to 45 rings
#   And then pick the closest one to 100,000 pixels total
def param_arr(n_lens,b_pxl,l_sys,detec_r,max_rad):
	tot_px = 100000.0
	if l_sys == 'Jiani3':
		scal_lens = 488.0/643.0
	elif l_sys == 'Sam1':
		scal_lens = 1.0
	arr,ix = [],[]
	for i in xrange(2,45):
		ix.append(i)
		arr.append(sum(curved_surface2(detec_r,2*max_rad,i,b_pxl)[2]))
	arr = np.asarray(arr)
	nstep = ix[np.argmin(np.absolute(arr*n_lens-tot_px))]
	t_px = arr[np.argmin(np.absolute(arr*n_lens-tot_px))]*n_lens
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
        while True:
                g = 0.000004         #0.1 for lns = 20, 0.002 for lns = 200, 0.000004 for lns = 1300
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

def display_configuration_from_array(config_name, config):
    n_lens = config[1]
    max_rad = config[2]  # Get's overwritten below
    b_pxl = config[6]
    print('=== Config: %s =====' % config_name)
    print ('  Detector radius:\t%0.2f'  % config[0])
    print ('  Number of lenses:\t%d'    % n_lens)
    print ('  Max radius:\t\t%0.2f'     % max_rad)
    print ('  EPD ratio:\t\t%0.2f'      % config[4])
    print ('  Number of rings (+1):\t%d' % config[5])
    print ('  Central pixels:\t%d'      % b_pxl)
    print ('  UUID:\t\t\t\t' + (str(config[8]) if len(config) > 8 else 'None'))
    print ('  Total pixels (in config):\t' + (str(config[7]) if len(config) > 7 else 'Not in config file'))

    lens_system_name = config_name.split('_')[0][2:]
    dtc_r = get_system_measurements(lens_system_name, max_rad)[1]
    n_step, tot_pxl = param_arr(n_lens, b_pxl, lens_system_name, dtc_r, max_rad)
    print ('  Total pixels (computed):\t%d'       % tot_pxl)
    # Focal length
    # vtx,max_rad,geom_eff = calc_rad(fibonacci_sphere(n_lens),sph_rad)   # This is what takes the time....
    # Cross check anything?
    # print ('  Geometric eff.:\t' + str(geom_eff))
    print('-------------------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Calculate distributed imaging detector geometry and configuration")
    parser.add_argument('lens_system_name', nargs='?', help='provide lens design (required unless "--list")')
    parser.add_argument('--list', '-l', action='store_true')
    args = parser.parse_args()
    lens_system_name = args.lens_system_name
    configs_pickle_file = '%sconf_file.p' % detector_pickled_path

    if args.list:
        import detectorconfig
        import pprint
        with open(configs_pickle_file, 'r') as f:
            dct = pickle.load(f)
        for key, value in dct.iteritems():
            display_configuration_from_array(key, value)
            dc = detectorconfig.DetectorConfig(value[0],
                                               value[1],
                                               value[2],
                                               value[3],
                                               0, 0, 1.0,
                                               lens_system_name='Sam1',
                                               EPD_ratio=value[4],
                                               light_confinement=True,
                                               nsteps=value[5],
                                               b_pixel=value[6],
                                               tot_pixels=value[7] if len(value) > 7 else None,
                                               uuid=value[8] if len(value) > 8 else None,
                                               config_name=key)
            pprint.pprint(vars(dc))

        print('========================')
        exit()

    if lens_system_name is None:
        parser.print_help()
        print
        print('Lens design is a required input argument (unless "--list" is used)')
        exit(-1)

    sph_rad = 7556.0
    b_pxl = int(raw_input('input number of pixels at the central ring: (more than 3) '))
    n_lens = int(raw_input('input the number of lens assemblies: '))
    EPD_ratio = float(raw_input('input the pupil ratio: '))
    vtx,max_rad,geom_eff = calc_rad(fibonacci_sphere(n_lens),sph_rad)
    dtc_r = get_system_measurements(lens_system_name,max_rad)[1]
    n_step,tot_pxl = param_arr(n_lens,b_pxl,lens_system_name,dtc_r,max_rad)
    conf_name = 'cf%s_K%i_%i'%(lens_system_name,n_lens,int(EPD_ratio*10))

    config = [sph_rad, n_lens, max_rad, vtx, EPD_ratio, n_step, b_pxl, tot_pxl, uuid.uuid1()]
    config_map_entry = {conf_name: config}
    display_configuration_from_array(conf_name, config)

    print '  Geometrical filling factor: %0.2f'%(geom_eff*math.pow(EPD_ratio,2))
    anw = raw_input('add %s configuration (y or n)?: '%conf_name)
    if anw == 'y':
        try:
            with open(configs_pickle_file,'r') as f:
                dct = pickle.load(f)
            if conf_name in dct:
                print('Replacing configuration: ' + conf_name)
            dct[conf_name] = config
            with open(configs_pickle_file,'w') as f:
                pickle.dump(dct,f,protocol=pickle.HIGHEST_PROTOCOL)
        except IOError:
            with open(configs_pickle_file,'w') as f:
                pickle.dump(config_map_entry, f, protocol=pickle.HIGHEST_PROTOCOL)
        print 'done'
    elif anw == 'n':
        exit()
    '''
    print 'rings in the optical system (+1): '+str(int(a[0]))
    print 'adjusted lens radius [mm]: '+str(int(a[1]))
    print 'total amount of pixels: '+str(int(sum(curved_surface2(dtc_r,2*max_rad,a[0],b_pxl)[2])*n_lens))
    '''
