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

def paralens(focal_length, diameter, nsteps=1024):
    #constructs a paraboloid lens
    height = 0.25/focal_length*(diameter/2)**2
    a = np.linspace(0, diameter/2, nsteps/2, endpoint=False)
    b = np.linspace(diameter/2, 0, nsteps/2)
    return make.rotate_extrude(np.concatenate((a, b)), np.concatenate((0.25/focal_length*(a)**2-height, -0.25/focal_length*(b)**2+height)), nsteps=64)

def cylindrical_shell(inner_radius, outer_radius, height, nsteps=1024):
    if inner_radius < outer_radius:
        return make.rotate_extrude([inner_radius, outer_radius, outer_radius, inner_radius], [-height/2.0, -height/2.0, height/2.0, height/2.0], nsteps)
    else:
        print "inner radius must be less than outer radius"

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
    from chroma.io.root import RootWriter
    import matplotlib.pyplot as plt

    #defining the specifications of the meshes
    lens3 = paralens(0.5, 1.0)
    
    #creating solids of form (mesh, inside_mat, outside_mat)
    lens_solid3 = Solid(lens3, lm.lensmat, lm.ls)
    
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

    #view(ftlneutrino)
    
    f = RootWriter('test.root')

    sim = Simulation(ftlneutrino)
   
    print "Data:"
    print "|z-coordinate | std | mean |"

    a=np.empty([11,3])
    k=0
    for ev in sim.simulate([photon_gauss((0.0, 5.0, z), 0.01, 1000000.0) for z in np.linspace(-1.0, 1.0, 11.0)], keep_photons_beg=True, keep_photons_end=True, run_daq=True, max_steps=100):

        detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)
      
        #f.write_event(ev) 

        a[k] = np.array([np.round(np.mean(ev.photons_beg.pos[detected][:,2]), decimals=1), np.std(ev.photons_end.pos[detected][:,2]), np.mean(ev.photons_end.pos[detected][:,2])])

        
        k = k+1
        print len(ev.photons_end[detected])
        

    print a
    

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(a[:,0], a[:,1])
    plt.xlabel('Initial Z position (m)')
    plt.ylabel('Standard Deviation of Final Z position')
    plt.title('Standard Deviation vs Initial position')

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(a[:,0], a[:,2])
    plt.xlabel('Mean Initial Z position (m)')
    plt.ylabel('Mean of Final Z position')
    plt.title('Mean of Final Position vs Initial Position')
    
    plt.show()

    '''
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(np.round(np.mean(ev.photons_beg.pos[detected][:,2]), decimals=1), np.std(ev.photons_end.pos[detected][:,2])) 
    plt.xlabel('Initial Z position (m)')
    plt.ylabel('Standard Deviation of Final Z position')
    plt.title('Standard Deviation vs Initial position')
    plt.show()
    '''

    f.close()

