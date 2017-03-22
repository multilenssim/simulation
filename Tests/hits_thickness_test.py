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

def klens(diameter, thickness, nsteps=1024):
    a = np.linspace(0, diameter/2, nsteps/2, endpoint=False)
    b = np.linspace(diameter/2, 0, nsteps/2)
    return make.rotate_extrude(np.concatenate((a, b)), np.concatenate((2*thickness/diameter**2*(a)**2-0.5*thickness, -2*thickness/diameter**2*(b)**2+0.5*thickness)), nsteps=64)

def cylindrical_shell(inner_radius, outer_radius, height, nsteps=1024):
    if inner_radius < outer_radius:
        return make.rotate_extrude([inner_radius, outer_radius, outer_radius, inner_radius], [-height/2.0, -height/2.0, height/2.0, height/2.0], nsteps)
    else:
        print "inner radius must be less than outer radius"

def uniform_circle(n, diameter, dir=(0,-1.0,0)):
    #constructs a uniform disk of photons.  See disk point picking @ wolfram.com.
    radii = np.random.uniform(0, diameter/2.0, n)
    angles = np.random.uniform(0, 2*np.pi, n)
    points = np.empty((n,3))
    points[:,0] = np.sqrt(diameter/2.0)*np.sqrt(radii)*np.cos(angles)
    points[:,1] = np.repeat(1.0, n)
    points[:,2] = np.sqrt(diameter/2.0)*np.sqrt(radii)*np.sin(angles)
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
    from chroma.io.root import RootWriter
    import matplotlib.pyplot as plt

    #defining the specifications of the mesh
    THICK = 0.5
    lensk = klens(1.0, THICK)

    def focal_length_finder(diameter, thickness):
        x = 0.25
        n = 1.5
        N = 2.0
        a = 2*thickness/diameter**2
        H = a/4*diameter**2
        u = np.arctan(2*a*x)
        m = -np.tan(np.pi/2-u+np.arcsin(n/N*np.sin(u)))
        b = a*x**2-m*x-H
        X = (m+np.sqrt(m**2+4*a*(-a*x**2+m*x+2*H)))/(-2*a)
        p = np.arctan((2*a*m*X-1)/(2*a*X+m))
        q = np.arcsin(N/n*np.sin(p))
        M = (1+2*a*X*np.tan(q))/(2*a*X-np.tan(q))
        focal_length = -a*X**2+H-M*X

        if b <= H:
            print "bad bad  bad bad bad"
        return focal_length
    
    #creating solid of form (mesh, inside_mat, outside_mat)
    lens_solidk = Solid(lensk, lm.lensmat, lm.ls)
    pmt = Solid(make.box(10,1,10), glass, lm.ls, surface=lm.fulldetect, color=0x00ff00)
    blocker = Solid(cylindrical_shell(0.5, 10.0, 0.000001), glass, lm.ls, surface=black_surface, color=0xff0000)

    #creates the entire Detector and adds the solid & pmt
    ftlneutrino = Detector(lm.ls)
    #ftlneutrino.add_solid(blocker)
    ftlneutrino.add_solid(lens_solidk)
    ftlneutrino.add_solid(lens_solidk, displacement=(0,0,2))
    ftlneutrino.add_pmt(pmt, displacement=(0,-0.5 - focal_length_finder(1.0, THICK), 0))
    #-2.02276 is the calculated value for paralens(0.5, 1.0)
    #-8.14654 is y value for lens1 with focal length 7.65654, box of sides: 1.0, R=5, d=1, T=0.0501256
    ftlneutrino.flatten()
    ftlneutrino.bvh = load_bvh(ftlneutrino)

    view(ftlneutrino)
    
    #f = RootWriter('test.root')

    sim = Simulation(ftlneutrino)
   
    uniformcir = uniform_circle(10000.0, 1.0)
    photongauss = photon_gauss((0.0, 10.0, 10.0), 0.01, 1000000.0)

    for ev in sim.simulate(photongauss, keep_photons_beg=True, keep_photons_end=True, run_daq=True, max_steps=100):

        detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)

        distance = np.mean(ev.photons_end.pos[detected][:,2])
        print distance
        print np.mean(ev.photons_end.pos[detected][:,0])
      
    #     #f.write_event(ev) 

    #     #prints standard deviation of ending x position, z position, and radius
    #     print np.std(ev.photons_end.pos[detected][:,0]), np.std(ev.photons_end.pos[detected][:,2]), np.std(np.sqrt((ev.photons_end.pos[detected][:,0])**2 + (ev.photons_end.pos[detected][:,2])**2))
 

        # fig = plt.figure(figsize=(7.8, 6))
        # plt.hist2d(ev.photons_beg.pos[detected][:,0], ev.photons_beg.pos[detected][:,2], bins=100)
        # plt.xlabel('Initial X-Position (m)')
        # plt.ylabel('Initial Z-Position (m)')
        # plt.title('Initial Position')
        # plt.colorbar()

        fig = plt.figure(figsize=(6, 6))
        plt.scatter(ev.photons_end.pos[detected][:,0], ev.photons_end.pos[detected][:,2])
        plt.xlabel('Final X-position (m)')
        plt.ylabel('Final Z-position (m)')
        plt.title('Final Location Plot')

        #range=[[-0.4, 0.4], [-0.4, 0.4]]
        
        fig = plt.figure(figsize=(7.8, 6))
        plt.hist2d(ev.photons_end.pos[detected][:,0], ev.photons_end.pos[detected][:,2], bins=100) 
        plt.xlabel('Final X-Position (m)')
        plt.ylabel('Final Z-Position (m)')
        plt.title('Hit Detection Locations')
        plt.colorbar()

        plt.show()
        
    #f.close()
