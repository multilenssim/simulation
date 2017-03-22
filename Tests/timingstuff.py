from chroma import make, view
from chroma.geometry import Geometry, Material, Solid, Surface
from chroma.transform import make_rotation_matrix
from chroma.demo.optics import glass, water, vacuum
from chroma.demo.optics import black_surface, r7081hqe_photocathode
from chroma.loader import create_geometry_from_obj
from chroma.detector import Detector
from chroma.pmt import build_pmt
from chroma.event import Photons
from chroma.sim import Simulation
from chroma.sample import uniform_sphere
import lensmaterials as lm
import numpy as np
#import pyparsing

def lens(radius, diameter, nsteps=1024):
    #constructs a convex lens mesh
    angles = np.linspace(-np.pi/2, np.pi/2, nsteps)
    return make.rotate_extrude(diameter/2*np.cos(angles), np.sign(angles)*(np.sqrt(radius**2-(diameter/2*np.cos(angles))**2)-np.sqrt(radius**2-(diameter/2)**2)), nsteps=64)

def pclens(radius, diameter, nsteps=1024):
    #constructs a plano-convex lens mesh
    angles = np.linspace(-np.pi/2, np.pi/2, nsteps)
    return make.rotate_extrude(diameter/2*np.cos(angles), (0.5*(np.sign(angles)+1))*(np.sqrt(radius**2-(diameter/2*np.cos(angles))**2)-np.sqrt(radius**2-(diameter/2)**2)), nsteps=64)

def paralens(focal_length, diameter, nsteps=1024):
    #constructs a paraboloid lens
    height = 0.25/focal_length*(diameter/2)**2
    a = np.linspace(0, diameter/2, nsteps/2, endpoint=False)
    b = np.linspace(diameter/2, 0, nsteps/2)
    return make.rotate_extrude(np.concatenate((a, b)), np.concatenate((0.25/focal_length*(a)**2-height, -0.25/focal_length*(b)**2+height)), nsteps=64)

def pcparalens(focal_length, diameter, nsteps=1024):
    #constructs a planoconvex paraboloid lens
    height = 0.25/focal_length*(diameter/2)**2
    a = np.linspace(0, diameter/2, nsteps/2, endpoint=False)
    b = np.linspace(diameter/2, 0, nsteps/2)
    c = np.tile(height, nsteps/2)
    return make.rotate_extrude(np.concatenate((a, b)), np.concatenate((0.25/focal_length*(a)**2, c)), nsteps=64)

def cylindrical_shell(inner_radius, outer_radius, height, nsteps=1024):
    if inner_radius < outer_radius:
        return make.rotate_extrude([inner_radius, outer_radius, outer_radius, inner_radius], [-height/2.0, -height/2.0, height/2.0, height/2.0], nsteps)
    else:
        print "inner radius must be less than outer radius"

def photon_rays(diameter, n):
    #constructs photons traveling in rays parallel with the optical axis of lens
    pos = np.array([(i, 1.0, 0) for i in np.linspace(-diameter/2, diameter/2, n)])
    dir = np.tile((0, -1.0, 0), (n,1))
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000, n)
    return Photons(pos, dir, pol, wavelengths)

def photon_rays_angledx(diameter, theta, n):
    #constructs collimated photons traveling at a specified angle from the x-axis in the xy-plane. z_initial = 0, y_initial = 1
    #theta ranges from -pi/2 to pi/2
    pos = np.array([(i+np.tan(theta), 1.0, 0) for i in np.linspace(-diameter/2, diameter/2, n)])
    dir = np.tile((-np.tan(theta),-1.0, 0), (n,1))
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000, n)
    return Photons(pos, dir, pol, wavelengths)

def photon_rays_angledx_circle(diameter, theta, n):
    #constructs collimated photons traveling in a uniform circle at a specified angle from the x-axis in the xy-plane. z_initial = 0, y_initial = 1
    #theta ranges from -pi/2 to pi/2
    radii = np.random.uniform(0, diameter/2, n)
    angles = np.random.uniform(0, 2*np.pi, n)
    points = np.empty((n,3))
    points[:,0] = np.sqrt(diameter/2)*np.sqrt(radii)*np.cos(angles) + np.tan(theta)
    points[:,1] = np.repeat(1.0, n)
    points[:,2] = np.sqrt(diameter/2)*np.sqrt(radii)*np.sin(angles)
    pos = points
    dir = np.tile((-np.tan(theta),-1,0), (n,1))
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000, n)
    return Photons(pos, dir, pol, wavelengths)

def photon_rays_angledz(diameter, theta, n):
    #constructs collimated photons traveling at a specified angle from the y axis in the yz plane. y-initial=1
    #theta ranges from -pi/2 to pi/2
    pos = np.array([(i, 1.0, np.tan(theta)) for i in np.linspace(-diameter/2, diameter/2, n)])
    dir = np.tile((0,-1.0,-np.tan(theta)), (n,1))
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000, n)
    return Photons(pos, dir, pol, wavelengths)

def photon_circle(n1, n2, diameter, dir=(0,-1,0)):
    '''constructs a circle of photons with higher density at the center
    this circle has repeat values at (0, 1, 0)
    for n=10,000: n1= 100, n2=100'''
    n = n1*n2
    pos = np.array([(r*np.cos(theta), 1.0, r*np.sin(theta)) for theta in np.linspace(0.0, 2*np.pi, n1, endpoint=False) for r in np.linspace(0.0, diameter/2, n2)])
    dir = np.tile(dir, (n,1))
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000, n)
    return Photons(pos, dir, pol, wavelengths)

def photon_circle2(n1, n2, diameter, dir=(0,-1,0)):
    '''constructs a circle of photons with higher density at the center
    this circle DOES NOT have repeat values at (0, 1, 0)
    for n=10,000: n1=101, n2=100'''
    n = n1*n2-n1+1
    def unique_rows(a):
        a = np.ascontiguousarray(a)
        unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
        return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
    pos = unique_rows(np.array([(r*np.cos(theta), 1.0, r*np.sin(theta)) for theta in np.linspace(0.0, 2*np.pi, n1, endpoint=False) for r in np.linspace(0.0, diameter/2, n2)]))
    dir = np.tile(dir, (n,1))
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000, n)
    return Photons(pos, dir, pol, wavelengths)

def uniform_circle(n, diameter, dir=(0,-1.0,0)):
    #constructs a uniform disk of photons.  See disk point picking @ wolfram.com.
    radii = np.random.uniform(0, diameter/2, n)
    angles = np.random.uniform(0, 2*np.pi, n)
    points = np.empty((n,3))
    points[:,0] = np.sqrt(diameter/2)*np.sqrt(radii)*np.cos(angles)
    points[:,1] = np.repeat(1.0, n)
    points[:,2] = np.sqrt(diameter/2)*np.sqrt(radii)*np.sin(angles)
    pos = points
    dir = np.tile(dir, (n,1))
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000, n)
    return Photons(pos, dir, pol, wavelengths)

def photon_gauss(pos, sigma, n):
    #constructs an initial distribution of photons with uniform angular position and radius determined by a normal distribution. Each photon is launched in a random direction.
    radii = np.random.normal(0.0, sigma, n)
    angles = np.linspace(0.0, np.pi, n, endpoint=False)
    points = np.empty((n,3))
    points[:,0] = radii*np.cos(angles) + pos[0]
    points[:,1] = np.tile(pos[1], n)
    points[:,2] = radii*np.sin(angles) + pos[2]
    pos = points
    dir = uniform_sphere(n)
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000.0, n)
    return Photons(pos, dir, pol, wavelengths) 

if __name__ == '__main__':
    from chroma.sim import Simulation
    from chroma import sample
    from chroma.event import Photons
    from chroma.loader import load_bvh
    from chroma.generator import vertex
    import matplotlib.pyplot as plt
    
    
    #defining the specifications of the meshes
    lens1 = lens(5.0, 1.0)
    lens2 = pclens(5.0, 1.0)
    lens3 = paralens(0.5, 1.0)
    lens4 = pcparalens(0.5,1.0)
    lens5 = make.box(1.0,1.0,1.0)
    
     #creating solids of form (mesh, inside_mat, outside_mat)
    lens_solid1 = Solid(lens1, lm.lensmat, lm.ls)
    lens_solid2 = Solid(lens2, lm.lensmat, lm.ls)
    lens_solid3 = Solid(lens3, lm.lensmat, lm.ls)
    lens_solid4 = Solid(lens4, lm.lensmat, lm.ls)
    lens_solid5 = Solid(lens5, lm.lensmat, lm.ls)
    
    pmt = Solid(make.box(10.0,1.0,10.0), glass, lm.ls, surface=lm.fulldetect)
    blocker = Solid(cylindrical_shell(0.5, 10.0, 0.000001), glass, lm.ls, surface=black_surface)

    
     #creates the entire Detector and adds the solid & pmt
    ftlneutrino = Detector(lm.ls)
    ftlneutrino.add_solid(blocker)
    ftlneutrino.add_solid(lens_solid3)
    ftlneutrino.add_pmt(pmt, rotation=None, displacement=(0,-2.02276, 0))
        #-2.02276 is the calculated value for paralens(0.5, 1.0)
        #-8.14654 is y value for lens1 with focal length 7.65654, box of sides: 1.0, R=5, d=1, T=0.0501256
    ftlneutrino.flatten()
    ftlneutrino.bvh = load_bvh(ftlneutrino)
    sim = Simulation(ftlneutrino)

    #view(ftlneutrino)
    
    #sim = Simulation(ftlneutrino)
   
    ftlphotons = photon_rays(1.0, 10000.0)
    ftlphotonsx = photon_rays_angledx(1.0, np.pi/4, 10000.0)
    ftlphotonsxcir = photon_rays_angledx_circle(1.0, np.pi/4, 10000.0)
    ftlphotonsz = photon_rays_angledz(1.0, np.pi/4, 10000.0)
    cirphotons = photon_circle(100.0, 100.0, 1.0)
    cirphotons2 = photon_circle2(101.0, 100.0, 1.0)
    uniformcir = uniform_circle(10000.0, 1.0)
    photongauss = photon_gauss((0.0, 10.0, 0.5), 0.01, 100000000.0)

    import timeit
    import time

    def timeforsim():
        sim.simulate(photongauss, keep_photons_beg=True, keep_photons_end=True, run_daq=True, max_steps=100)
        
        
    x = timeit.timeit(stmt='timeforsim()', setup='from __main__ import timeforsim', number=10)

    z = timeit.timeit(stmt='timeforsim()', setup='from __main__ import timeforsim', number=10)

    w = timeit.timeit(stmt='timeforsim()', setup='from __main__ import timeforsim', number=10)

    w2 = timeit.timeit(stmt='timeforsim()', setup='from __main__ import timeforsim', number=10)
     
    print x, z, w, w2
    
    '''y = timeit.timeit(stmt='happytimes()', setup='from __main__ import happytimes', number=10)
    print y'''
