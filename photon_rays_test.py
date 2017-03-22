import numpy as np
def photon_rays_angledx(n, diameter, theta):
    pos = np.array([(i+np.tan(theta), 1.0, 0) for i in np.linspace(-diameter/2.0, diameter/2.0, n)])
    print pos
    dir = np.tile((-np.sin(theta), -np.cos(theta), 0), (n, 1))
    return dir
    
print photon_rays_angledx(20, 1.0, 0)
print 2.7/1.8
print 1.2*1.5