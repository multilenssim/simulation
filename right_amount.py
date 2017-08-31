from lenssystem import get_system_measurements,get_half_EPD
import numpy as np


def calc_steps(x_value,y_value,detector_r,base_pixel):
        x_coord = np.asarray([x_value,np.roll(x_value,-1)]).T[:-1]
        y_coord = np.asarray([y_value,np.roll(y_value,-1)]).T[:-1]
        lat_area = 2*np.pi*detector_r*(y_coord[:,0]-y_coord[:,1])
        n_step = (lat_area/lat_area[-1]*base_pixel).astype(int)
        return x_coord, y_coord, n_step

def curved_surface2(detector_r=2.0, diameter = 2.5, nsteps=20,base_pxl=4,ret_arr=False):
    '''Builds a curved surface based on the specified radius. Origin is center of surface.'''
    if (detector_r < diameter/2.0):
        raise Exception('The Radius of the curved surface must be larger than diameter/2.0')
    shift1 = np.sqrt(detector_r**2 - (diameter/2.0)**2)
    theta1 = np.arctan(shift1/(diameter/2.0))
    angles1 = np.linspace(theta1, np.pi/2, nsteps)
    x_value = abs(detector_r*np.cos(angles1))
    y_value = detector_r-detector_r*np.sin(angles1)
    surf = None
    x_coord,y_coord,n_step = calc_steps(x_value,y_value,detector_r,base_pixel=base_pxl)
    return calc_steps(x_value,y_value,detector_r,base_pxl)

def param_arr(base,b_pxl,l_sys):
	if l_sys == 'Jiani3':
		scal_lens = 488.0/643.0
	elif l_sys == 'Sam1':
		scal_lens = 1.0
	arr,ix = [],[]
	for i in xrange(2,20):
		ix.append(i)
		arr.append(np.cumsum(curved_surface2(dtc_r,2*max_rad,i,b_pxl)[2])[-1])
	arr = np.asarray(arr)
	dct = np.stack((ix,scal_lens*10000/(2*(np.sqrt(10000/arr).astype(int)+np.sqrt(3)-1))))
	sel_arr = np.absolute(((base*(base+1))/2*arr-5000)/5000.0)<0.1
	dct = dct[:,sel_arr]
	if dct.shape[1]>1:
		print dct
		dct = dct[:,np.argmin(np.absolute((base*(base+1))/2*arr[sel_arr]-5000))]
	return dct

if __name__ == '__main__':
	lens_system_name = 'Jiani3'
	b_pxl = int(raw_input('input number of pixels at the central ring: (more than 3) '))
	base = int(raw_input('input the number of optical system at the base: '))
	max_rad = 10000.0/(2*(base+np.sqrt(3)-1))
	dtc_r = get_system_measurements(lens_system_name,max_rad)[1]
	a = param_arr(base,b_pxl,lens_system_name)
	print 'rings in the optical system (+1): '+str(int(a[0]))
	print 'lenses per face: '+str(int((base*(base+1))/2))
	print 'adjusted lens radius [mm]: '+str(int(a[1]))
	print 'total amount of pixels: '+str(int(sum(curved_surface2(dtc_r,2*max_rad,a[0],b_pxl)[2])*(base*(base+1))/2*20))
