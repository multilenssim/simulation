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

def param_arr(rad_lens,b_pxl,l_sys):
	if l_sys == 'Jiani3':
		scal_lens = 488.0/643.0
	elif l_sys == 'Sam1':
		scal_lens = 1.0
	arr,ix = [],[]
	for i in xrange(2,20):
		ix.append(i)
		arr.append(np.cumsum(curved_surface2(detector_r=2.0, diameter = 2.5, nsteps=i,base_pxl=b_pxl)[2])[-1])
	dct = np.stack((ix,np.sqrt(10000/np.asarray(arr)).astype(int),scal_lens*10000/(2*(np.sqrt(10000/np.asarray(arr)).astype(int)+np.sqrt(3)-1))))
	dct = dct[:,np.absolute((dct[1]*(dct[1]+1)/2*arr*20-100000)/100000.0)<0.1]
	iex = np.absolute(dct[2]-rad_lens).argmin()
	return dct[:,iex]

if __name__ == '__main__':
	b_pxl = 4	
	a = param_arr(450.0,b_pxl,'Jiani3')
	print 'rings in the optical system (+1): '+str(int(a[0]))
	print 'base lenses: '+str(int(a[1]))+' lenses per face: '+str(int((a[1]*(a[1]+1))/2))
	print 'adjusted lens radius [mm]: '+str(int(a[2]))
	print 'total amount of pixels: '+str(int(sum(curved_surface2(detector_r=2.0, diameter = 2.5, nsteps=a[0] ,base_pxl=b_pxl,ret_arr=False)[2])*(a[1]*(a[1]+1))/2*20))
