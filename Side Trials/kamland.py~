from chroma import make, view
from chroma.geometry import Geometry, Material, Mesh, Solid, Surface
from chroma.demo.optics import glass, black_surface
from chroma.detector import Detector
from chroma.pmt import build_pmt
from chroma.event import Photons
from chroma.sim import Simulation
from chroma.sample import uniform_sphere
from chroma.transform import make_rotation_matrix
import lensmaterials as lm
import numpy as np

def paralens(focal_length, diameter, nsteps=128):
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

lens = Solid(paralens(0.5, 1.0), lm.lensmat, lm.ls)
#blocker = Solid(blockermesh(1.0), glass, lm.ls, surface=black_surface)


def find_triangle_centers(E, n):
    """input edge length of icosahedron 'E' and number of small triangles in the base of each face 'n' to create an array of the coordinates of the centroid of each small triangle."""
    L = E/n
    R = L/(2.0*np.sqrt(3.0))
    print L
    print R
    
    """First, a list of index coordinates are created for each triangle in the side. These are then transformed into the actual coordinate positions of the centers of the triangles based on the relevant dimensions.
    """
    x = np.linspace(0, 2*n-2, n)
    rep = np.linspace(2*n-1, 1, n)
    partial_index = np.linspace(1, n-1, n-1)
    xindices = np.repeat(x[0], rep[0])
    yindices = np.linspace(0, 2*n-2, 2*n-1)
    for i in partial_index:
        xindices = np.concatenate((xindices, np.repeat(x[i],rep[i])))
        yindices = np.concatenate((yindices, np.linspace(0, 2*(n-i-1), 2*(n-i)-1)))
    print xindices
    print yindices
    xindices = xindices + np.ceil(yindices/2.0)
    print xindices

    xcoords = L/2.0*(xindices+1)
    ycoords = R*np.floor(0.5*(3*yindices+2))
    print xcoords
    print ycoords
    totalshiftx = -E/2.0
    totalshifty = -E/(2.0*np.sqrt(3.0))
    print xcoords + totalshiftx, ycoords + totalshifty
    xhaps = xcoords + totalshiftx
    yhaps = ycoords + totalshifty

    newkamland = Detector(lm.ls)
    for i in np.linspace(0, n**2-1, n**2):
        newkamland.add_solid(Solid(paralens(1.0, 0.5), lm.lensmat, lm.ls), rotation=make_rotation_matrix(np.pi/2, (1,0,0)), displacement= (xhaps[i], yhaps[i], 0))
    view(newkamland)
    



if __name__ == '__main__':
    from chroma import sample
    from chroma.generator import vertex
    import matplotlib.pyplot as plt

    find_triangle_centers(10.0*np.sqrt(3.0), 10)
   
    #newkamland = Detector(lm.ls)
    #for i in xhaps:
       # newkamland.add_solid(Solid(paralens(1.0, 0.5), lm.lensmat, lm.ls), rotation=None, displacement= (xhaps[i], yhaps[i], 0))
    #view(newkamland)
    
