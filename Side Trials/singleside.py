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

def lens(diameter, thickness, nsteps=128):
    a = np.linspace(0, diameter/2, nsteps/2, endpoint=False)
    b = np.linspace(diameter/2, 0, nsteps/2)
    return make.rotate_extrude(np.concatenate((a, b)), np.concatenate((2*thickness/diameter**2*(a)**2-0.5*thickness, -2*thickness/diameter**2*(b)**2+0.5*thickness)), nsteps=128)

def cylindrical_shell(inner_radius, outer_radius, thickness, nsteps=128):
    #make sure that nsteps is the same as that of rotate extrude in lens
    if inner_radius < outer_radius:
        return make.rotate_extrude([inner_radius, outer_radius, outer_radius, inner_radius], [-thickness/2.0, -thickness/2.0, thickness/2.0, thickness/2.0], nsteps)
    else:
        print "inner radius must be less than outer radius"

def inner_blocker(radius, thickness, n):
    #input radius of circles to create blocker shape for a third of the space inside of three tangent circles
    h = np.sqrt(3)/12*radius
    a = np.linspace(-radius/2.0, radius/2.0, 4, endpoint=False)
    b = np.linspace(radius/2.0, -radius/2.0, n, endpoint=False)
    y = np.concatenate((np.linspace(h, -h, 2, endpoint=False), np.linspace(-h, h, 2, endpoint=False), -np.sqrt(radius**2-b**2)+7*np.sqrt(3)/12*radius))
    return make.linear_extrude(np.concatenate((a,b)), y, thickness)

def outer_blocker(radius, thickness):
    #constructs trapezoidal blockers for the perimeter of the side
    return make.linear_extrude([-radius/np.sqrt(3), -2/np.sqrt(3)*radius, 2/np.sqrt(3)*radius, radius/np.sqrt(3)], [0, -radius, -radius, 0], thickness)

def corner_blocker(radius, thickness):
    #constructs triangle corners for a side
    return make.linear_extrude([0, -radius/np.sqrt(3), radius/np.sqrt(3)], [2.0/3*radius, -radius/3.0, -radius/3.0], thickness)
    
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

def build_side(E, n, d, t, blockers=True, b=1.0/1000):
    """input edge length of icosahedron 'E', the number of small triangles in the base of each face 'n', the ratio of the diameter of each lens to the maximum diameter possible 'd', the ratio of the thickness of the lens to the chosen diameter 't', and the ratio of the thickness of the blockers to that of the lenses 'b' to return a face of kabamland. First, a list of index coordinates are created for each hexagon within a tiled triangle. These are then transformed into the actual coordinate positions for the lenses based on the parameters.
    """
    R = E/(2*np.sqrt(3)*n)
    key = np.empty(3*n-2)
    for i in np.linspace(0, 3*(n-1), 3*n-2):
        key[i] = n-i+2*np.floor(i/3)
    xindices = np.linspace(0, 2*(n-1), n)
    yindices = np.repeat(0,n)
    for i in np.linspace(1, 3*(n-1), 3*(n-1)):
        xindices = np.concatenate((xindices, np.linspace(n-key[i], n+key[i]-2, key[i])))
        yindices = np.concatenate((yindices, np.repeat(i,key[i])))
    xcoords = E/(2.0*n)*(xindices+1)-E/2.0
    ycoords = R*(yindices+1)-E/(2*np.sqrt(3))
    
    #viewing the full detector
    kabamland = Detector(lm.ls)
    #kabamland.add_solid(Solid(inner_blocker(R, 128), lm.lensmat, lm.ls, black_surface, 0xff0000))
    #kabamland.add_solid(Solid(outer_blocker(R), lm.lensmat, lm.ls, black_surface, 0xff0000))
    #kabamland.add_solid(Solid(corner_blocker(R), lm.lensmat, lm.ls, black_surface, 0xff0000))
    for i in np.linspace(0, 3*n*(n-1)/2, (3*n**2-3*n+2)/2):
        kabamland.add_solid(Solid(lens(2*d*R, 2*d*R*t), lm.lensmat, lm.ls), rotation=make_rotation_matrix(np.pi/2, (1, 0, 0)), displacement=(xcoords[i], ycoords[i], 0))
    if blockers:
        blocker_thickness = 2*R*d*t*b
        for x in np.linspace(1, 2*n-3, n-1):
            kabamland.add_solid(Solid(outer_blocker(R, blocker_thickness), lm.lensmat, lm.ls, black_surface, 0xff0000), displacement=(E/(2.0*n)*(x+1)-E/2.0, R-E/(2*np.sqrt(3)), 0))
            kabamland.add_solid(Solid(outer_blocker(R, blocker_thickness), lm.lensmat, lm.ls, black_surface, 0xff0000), rotation=make_rotation_matrix(2*np.pi/3, (0, 0, 1)), displacement=(E/(2.0*n)*(0.5*x+1)-E/2.0, R*(3/2.0*x+1)-E/(2*np.sqrt(3)), 0))
            kabamland.add_solid(Solid(outer_blocker(R, blocker_thickness), lm.lensmat, lm.ls, black_surface, 0xff0000), rotation=make_rotation_matrix(4*np.pi/3, (0, 0, 1)), displacement=(E/(2.0*n)*(-0.5*x+2*n-1)-E/2.0, R*(3/2.0*x+1)-E/(2*np.sqrt(3)), 0))
        for theta in np.linspace(3*np.pi/6, 11*np.pi/6, 3):
            kabamland.add_solid(Solid(corner_blocker(R, blocker_thickness), lm.lensmat, lm.ls, black_surface, 0xff0000), displacement=(2*R*(n-1/3.0)*np.cos(theta), 2*R*(n-1/3.0)*np.sin(theta), 0))
        for i in np.linspace(0, 3*n*(n-1)/2, (3*n**2-3*n+2)/2):
            for j in np.linspace(1, 11, 6): 
                kabamland.add_solid(Solid(inner_blocker(R, blocker_thickness, 128), lm.lensmat, lm.ls, black_surface, 0xff0000), rotation=make_rotation_matrix(j*np.pi/6, (0, 0, 1)), displacement=(xcoords[i] + 7*np.sqrt(3)/12*R*np.cos(3*np.pi/2-j*np.pi/6), ycoords[i]+7*np.sqrt(3)/12*R*np.sin(3*np.pi/2-j*np.pi/6), 0))
            if d < 1:
                kabamland.add_solid(Solid(cylindrical_shell(d*R, R, blocker_thickness), lm.lensmat, lm.ls, black_surface, 0xff0000), rotation=make_rotation_matrix(np.pi/2, (1,0,0)), displacement=(xcoords[i], ycoords[i], 0))
    
    view(kabamland)
    
if __name__ == '__main__':
    from chroma import sample
    from chroma.generator import vertex
    import matplotlib.pyplot as plt

    build_side(10, 3, 0.8, 0.1)
   
