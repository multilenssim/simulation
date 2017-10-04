from lenssystem import get_system_measurements,get_half_EPD
import numpy as np
import argparse


def calc_steps(x_value,y_value,detector_r,base_pixel):
        x_coord = np.asarray([x_value,np.roll(x_value,-1)]).T[:-1]
        y_coord = np.asarray([y_value,np.roll(y_value,-1)]).T[:-1]
        lat_area = 2*np.pi*detector_r*(y_coord[:,0]-y_coord[:,1])
        n_step = (lat_area/lat_area[-1]*base_pixel).astype(int)
        return x_coord, y_coord, n_step

def curved_surface2(detector_r=2.0, diameter = 2.5, nsteps=20,base_pxl=4):
    '''Builds a curved surface based on the specified radius. Origin is center of surface.'''
    if (detector_r < diameter/2.0):
        raise Exception('The Radius of the curved surface must be larger than diameter/2.0')
    shift1 = np.sqrt(detector_r**2 - (diameter/2.0)**2)
    theta1 = np.arctan(shift1/(diameter/2.0))
    angles1 = np.linspace(theta1, np.pi/2, nsteps)
    x_value = abs(detector_r*np.cos(angles1))
    y_value = detector_r-detector_r*np.sin(angles1)
    x_coord,y_coord,n_step = calc_steps(x_value,y_value,detector_r,base_pixel=base_pxl)
    return calc_steps(x_value,y_value,detector_r,base_pxl)

def param_arr(base,b_pxl,l_sys,detec_r,max_rad):
	edge_len = 10000.0
	px_per_face = 5000.0
	if l_sys == 'Jiani3':
		scal_lens = 488.0/643.0
	elif l_sys == 'Sam1':
		scal_lens = 1.0
	arr,ix = [],[]
	for i in xrange(2,45):
		ix.append(i)
		arr.append(sum(curved_surface2(detec_r,2*max_rad,i,b_pxl)[2]))
	arr = np.asarray(arr)
	dct = np.stack((ix,scal_lens*edge_len/(2*(np.sqrt(2*px_per_face/arr).astype(int)+np.sqrt(3)-1))))
	print arr
	sel_arr = np.absolute(((base*(base+1))/2*arr-px_per_face)/px_per_face)<0.3
	dct = dct[:,sel_arr]
	if dct.shape[1]>1:
		dct = dct[:,np.argmin(np.absolute((base*(base+1))/2*arr[sel_arr]-5000))]
	return dct

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('lens_system_name',help='provide lens design')
	args = parser.parse_args()
	lens_system_name = args.lens_system_name
	b_pxl = int(raw_input('input number of pixels at the central ring: (more than 3) '))
	base = int(raw_input('input the number of optical system at the base: '))
	max_rad = 10000.0/(2*(base+np.sqrt(3)-1))
	dtc_r = get_system_measurements(lens_system_name,max_rad)[1]
	a = param_arr(base,b_pxl,lens_system_name,dtc_r,max_rad)
	print 'rings in the optical system (+1): '+str(int(a[0]))
	print 'lenses per face: '+str(int((base*(base+1))/2))
	print 'adjusted lens radius [mm]: '+str(int(a[1]))
	print 'total amount of pixels: '+str(int(sum(curved_surface2(dtc_r,2*max_rad,a[0],b_pxl)[2])*(base*(base+1))/2*20))
