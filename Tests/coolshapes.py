'''
from chroma import make, view
from chroma.geometry import Geometry, Material, Mesh, Solid, Surface
from chroma.demo.optics import glass, black_surface
from chroma.detector import Detector
from chroma.pmt import build_pmt
from chroma.event import Photons
from chroma.sim import Simulation
from chroma.sample import uniform_sphere
import lensmaterials as lm
import numpy as np

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

def blocker_mesh(diameter, outer_width, thickness, nsteps=66):
    """builds a triangular blocker with a hole of specified diameter, outer width, and thickness.  Outer width is shortest distance from hole to perimeter of triangle
    s = side length of triangle"""
    angles = np.linspace(np.pi/6, 13*np.pi/6, nsteps, endpoint=False)
    angles2 = np.linspace(np.pi/6, -11*np.pi/6, nsteps, endpoint=False)
    s = np.sqrt(3)*(diameter+2*outer_width)
    a = s/(2*np.sqrt(3)*np.cos(np.linspace(np.pi/6, np.pi/2, nsteps/6, endpoint=False)-np.pi/6))
    b = s/(2*np.sqrt(3)*np.cos(5*np.pi/6-np.linspace(np.pi/2, 5*np.pi/6, nsteps/6, endpoint=False)))
    
    triangle_radii = np.tile(np.concatenate((a, b)), 3)

    circle = make.linear_extrude(diameter/2*np.cos(angles), diameter/2*np.sin(angles), thickness)
    
    triangle = make.linear_extrude(triangle_radii*np.cos(angles), triangle_radii*np.sin(angles), thickness)
    
    x_coords = np.resize(np.dstack((triangle_radii*np.cos(angles), diameter/2*np.cos(angles))), (2*nsteps))
    y_coords = np.resize(np.dstack((triangle_radii*np.sin(angles), diameter/2*np.sin(angles))), (2*nsteps))
    fullthing = make.linear_extrude(x_coords, y_coords, thickness)

    xco2 = np.concatenate((triangle_radii*np.cos(angles), diameter/2*np.cos(angles2)))
    yco2 = np.concatenate((triangle_radii*np.sin(angles), diameter/2*np.sin(angles2)))
    fullthing2 = make.linear_extrude(xco2, yco2, thickness)
    
    xco3 = np.resize(np.dstack((triangle_radii*np.cos(angles), diameter/2*np.cos(angles), triangle_radii*np.cos(angles))), (3*nsteps))
    yco3 = np.resize(np.dstack((triangle_radii*np.sin(angles), diameter/2*np.sin(angles), triangle_radii*np.sin(angles))), (3*nsteps))
    fullthing3 = make.linear_extrude(xco3, yco3, thickness)

    xco4 = diameter/2*np.cos(angles)
    yco4 = diameter/2*np.sin(angles)
    x2co4 = triangle_radii*np.cos(angles)
    y2co4 = triangle_radii*np.sin(angles)
    fullthing4 = make.linear_extrude(xco4, yco4, thickness, x2co4, y2co4) 

    return fullthing
    
    #L = np.repeat(thickness/2.0, nsteps)

    #fullthing5 = make.rotate_extrude([np.repeat(diameter/2,nsteps), triangle_radii, triangle_radii, np.repeat(diameter/2,nsteps)], [-L, -L, L, L], nsteps)


    c = diameter/2*np.cos(angles)
    d = triangle_radii*np.cos(angles)
    e = diameter/2*np.sin(angles)
    f = triangle_radii*np.sin(angles)
    points = np.empty((4*nsteps,3))
    points[:,0] = np.resize(np.dstack((c, d, d, c)), 4*nsteps)
    points[:,1] = np.resize(np.dstack((e, f, f, e)), 4*nsteps)
    points[:,2] = np.tile((thickness/2, thickness/2, -thickness/2, -thickness/2), nsteps)
    
    return Mesh(points, np.resize(np.tile((1,1,1), 100), (100, 3)), remove_duplicate_vertices=True)'''

    #happy = triangle.__add__(circle)
    #return happy

'''def photon_gauss(pos, sigma, n):
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

def lens_with_blocker(lens):
    #builds a lens inside of a triangular blocker
    d = Detector(lm.ls)
    

if __name__ == '__main__':
    from chroma import sample
    from chroma.generator import vertex
    import matplotlib.pyplot as plt

    print blocker_mesh(1.0,0.2,0.1).get_triangle_centers()

    apple = Solid(blocker_mesh(1.0, 0.2, 0.1), lm.lensmat, lm.ls)
    newkamland = Detector(lm.ls)
    newkamland.add_solid(apple)
    view(apple)


'''

from chroma import make, view
from chroma.geometry import Geometry, Material, Mesh, Solid, Surface
from chroma.demo.optics import glass, black_surface
from chroma.detector import Detector
from chroma.pmt import build_pmt
from chroma.event import Photons
from chroma.sim import Simulation
from chroma.sample import uniform_sphere
from chroma.transform import make_rotation_matrix, normalize
import meshhelper as mh
import lensmaterials as lm
import numpy as np

def lens(diameter, thickness, nsteps=16):
    #constructs a parabolic lens
    a = np.linspace(0, diameter/2, nsteps/2, endpoint=False)
    b = np.linspace(diameter/2, 0, nsteps/2)
    return make.rotate_extrude(np.concatenate((a, b)), np.concatenate((2*thickness/diameter**2*(a)**2-0.5*thickness, -2*thickness/diameter**2*(b)**2+0.5*thickness)), nsteps=16)

def cylindrical_shell(inner_radius, outer_radius, thickness, nsteps=16):
    #make sure that nsteps is the same as that of rotate extrude in lens
    if inner_radius < outer_radius:
        return make.rotate_extrude([inner_radius, outer_radius, outer_radius, inner_radius], [-thickness/2.0, -thickness/2.0, thickness/2.0, thickness/2.0], nsteps)
    else:
        print "inner radius must be less than outer radius"

def inner_blocker(radius, thickness, n=128):
    #input radius of circles to create blocker shpae for a third of the space between three tangent circles
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

def return_values(edge_length, base):
        edge_length = float(edge_length)
        max_radius = edge_length/(2*np.sqrt(3)*base)
        phi = (1+np.sqrt(5))/2

        #lists of the coordinate centers of each face and vertices of the icosahedron
        facecoords = np.array([[phi**2/6*edge_length, phi**2/6*edge_length, phi**2/6*edge_length], [phi**2/6*edge_length, phi**2/6*edge_length, -phi**2/6*edge_length], [phi**2/6*edge_length, -phi**2/6*edge_length, phi**2/6*edge_length], [phi**2/6*edge_length, -phi**2/6*edge_length, -phi**2/6*edge_length], [-phi**2/6*edge_length, phi**2/6*edge_length, phi**2/6*edge_length], [-phi**2/6*edge_length, phi**2/6*edge_length, -phi**2/6*edge_length], [-phi**2/6*edge_length, -phi**2/6*edge_length, phi**2/6*edge_length], [-phi**2/6*edge_length, -phi**2/6*edge_length, -phi**2/6*edge_length], [0.0, edge_length*phi/6, edge_length*(2*phi+1)/6], [0.0, edge_length*phi/6, -edge_length*(2*phi+1)/6], [0.0, -edge_length*phi/6, edge_length*(2*phi+1)/6], [0.0, -edge_length*phi/6, -edge_length*(2*phi+1)/6], [edge_length*phi/6, edge_length*(2*phi+1)/6, 0.0], [edge_length*phi/6, -edge_length*(2*phi+1)/6, 0.0], [-edge_length*phi/6, edge_length*(2*phi+1)/6, 0.0], [-edge_length*phi/6, -edge_length*(2*phi+1)/6, 0.0], [edge_length*(2*phi+1)/6, 0.0, edge_length*phi/6], [edge_length*(2*phi+1)/6, 0.0, -edge_length*phi/6], [-edge_length*(2*phi+1)/6, 0.0, edge_length*phi/6], [-edge_length*(2*phi+1)/6, 0.0, -edge_length*phi/6]])

        vertices = np.array([[edge_length*phi/2, edge_length/2, 0], [edge_length*phi/2, edge_length/2, 0], [edge_length*phi/2, -edge_length/2, 0], [edge_length*phi/2, -edge_length/2, 0], [-edge_length*phi/2, edge_length/2, 0], [-edge_length*phi/2, edge_length/2, 0], [-edge_length*phi/2, -edge_length/2, 0], [-edge_length*phi/2, -edge_length/2, 0], [-edge_length/2, 0, edge_length*phi/2], [edge_length/2, 0, -edge_length*phi/2], [-edge_length/2, 0, edge_length*phi/2], [edge_length/2, 0, -edge_length*phi/2], [edge_length*phi/2, edge_length/2, 0], [edge_length*phi/2, -edge_length/2, 0], [-edge_length*phi/2, edge_length/2, 0], [-edge_length*phi/2, -edge_length/2, 0], [edge_length*phi/2, edge_length/2, 0], [edge_length*phi/2, -edge_length/2, 0], [-edge_length*phi/2, edge_length/2, 0], [-edge_length*phi/2, -edge_length/2, 0]])

        #rotating each face onto the plane perpindicular to a line from the origin to the center of the face.
        direction = -normalize(facecoords)
        axis = np.cross(direction, np.array([0.0, 0.0, 1.0]))
        angle = np.arccos(direction[:,2])

        A = np.empty((20, 3))
        B = np.empty((20, 3))
        spin_sign = np.empty(20)
        spin_angle = np.empty(20)
        for k in range(20):
            A[k] = np.dot(make_rotation_matrix(angle[k], axis[k]), np.array([0, edge_length/np.sqrt(3), 0]))
            B[k] = vertices[k] - facecoords[k]
            spin_sign[k] = np.sign(np.dot(np.dot(A[k], make_rotation_matrix(np.pi/2, facecoords[k])), B[k]))
            spin_angle[k] = spin_sign[k]*np.arccos(3*np.dot(A[k], B[k])/edge_length**2)
        
        return edge_length, max_radius, phi, facecoords, vertices, direction, axis, angle, spin_angle

def build_lens_icosahedron(edge_length, base, diameter_ratio, thickness_ratio, blockers=True, blocker_thickness_ratio=1.0/1000):
    '''input edge length of icosahedron 'edge_length', the number of small triangles in the base of each face 'base', the ratio of the diameter of each lens to the maximum diameter possible 'diameter_ratio', the ratio of the thickness of the lens to the chosen (not maximum) diameter 'thickness_ratio', and the ratio of the thickness of the blockers to that of the lenses 'blocker_thickness_ratio' to return the icosahedron of lenses in kabamland.'''
    
    edge_length, max_radius, phi, facecoords, vertices, direction, axis, angle, spin_angle = return_values(edge_length, base)

    #iterating the lenses into a hexagonal pattern within a single side. First coordinate indices are created, and then these are transformed into the actual coordinate positions based on the parameters given.
    key = np.empty(3*base-2)
    for i in range(3*base-2):
        key[i] = base-i+2*np.floor(i/3)
    xindices = np.linspace(0, 2*(base-1), base)
    yindices = np.repeat(0,base)
    for i in np.linspace(1, 3*(base-1), 3*(base-1)):
        xindices = np.concatenate((xindices, np.linspace(base-key[i], base+key[i]-2, key[i])))
        yindices = np.concatenate((yindices, np.repeat(i,key[i])))
    xcoords = edge_length/(2.0*base)*(xindices+1)-edge_length/2.0
    ycoords = max_radius*(yindices+1)-edge_length/(2*np.sqrt(3))

    #creating the lenses for a single face
    initial_lens = mh.rotate(lens(2*diameter_ratio*max_radius, 2*diameter_ratio*max_radius*thickness_ratio), make_rotation_matrix(np.pi/2, (1,0,0)))
    face = Solid(mh.shift(initial_lens, (xcoords[0], ycoords[0], 0)), lm.lensmat, lm.ls) 
    for i in np.linspace(1, 3*base*(base-1)/2, (3*base**2-3*base+2)/2-1):
        face = face + Solid(mh.shift(initial_lens, (xcoords[i], ycoords[i], 0)), lm.lensmat, lm.ls) 
    
    #creating the various blocker shapes to fill in the empty space of a single face.
    if blockers:
        blocker_thickness = 2*max_radius*diameter_ratio*thickness_ratio*blocker_thickness_ratio
 
        for theta in np.linspace(3*np.pi/6, 11*np.pi/6, 3):
            face = face + Solid(mh.shift(corner_blocker(max_radius, blocker_thickness), (2*max_radius*(base-1/3.0)*np.cos(theta), 2*max_radius*(base-1/3.0)*np.sin(theta), 0)), lm.lensmat, lm.ls, black_surface, 0xff0000)

        for x in np.linspace(1, 2*base-3, base-1):
            face = face + Solid(mh.shift(outer_blocker(max_radius, blocker_thickness), (edge_length/(2*base)*(x+1)-edge_length/2, max_radius-edge_length/(2*np.sqrt(3)), 0)),  lm.lensmat, lm.ls, black_surface, 0xff0000) + Solid(mh.shift(mh.rotate(outer_blocker(max_radius, blocker_thickness), make_rotation_matrix(2*np.pi/3, (0, 0, 1))), (edge_length/(2*base)*(0.5*x+1)-edge_length/2, max_radius*(3/2.0*x+1)-edge_length/(2*np.sqrt(3)), 0)), lm.lensmat, lm.ls, black_surface, 0xff0000) + Solid(mh.shift(mh.rotate(outer_blocker(max_radius, blocker_thickness), make_rotation_matrix(4*np.pi/3, (0, 0, 1))), (edge_length/(2*base)*(-0.5*x+2*base-1)-edge_length/2, max_radius*(3/2.0*x+1)-edge_length/(2*np.sqrt(3)), 0)),  lm.lensmat, lm.ls, black_surface, 0xff0000)

        for i in np.linspace(0, 3*base*(base-1)/2, (3*base**2-3*base+2)/2):
            for j in np.linspace(1, 11, 6): 
                face = face + Solid(mh.shift(mh.rotate(inner_blocker(max_radius, blocker_thickness, 128), make_rotation_matrix(j*np.pi/6, (0, 0, 1))), (xcoords[i] + 7*np.sqrt(3)/12*max_radius*np.cos(3*np.pi/2-j*np.pi/6), ycoords[i]+7*np.sqrt(3)/12*max_radius*np.sin(3*np.pi/2-j*np.pi/6), 0)), lm.lensmat, lm.ls, black_surface, 0xff0000)

            if diameter_ratio < 1:
                face = face + Solid(mh.shift(mh.rotate(cylindrical_shell(diameter_ratio*max_radius, max_radius, blocker_thickness), make_rotation_matrix(np.pi/2, (1,0,0))), (xcoords[i], ycoords[i], 0)), lm.lensmat, lm.ls, black_surface, 0xff0000)

    #creating all 20 faces and putting them into the detector with the correct orientations.
    for k in range(20):   
        kabamland.add_solid(face, rotation=np.dot(make_rotation_matrix(spin_angle[k], direction[k]), make_rotation_matrix(angle[k], axis[k])), displacement=facecoords[k])

def build_pmt_icosahedron(edge_length, base, diameter_ratio, thickness_ratio, blockers=True, blocker_thickness_ratio=1.0/1000):
    edge_length, max_radius, phi, facecoords, vertices, direction, axis, angle, spin_angle = return_values(edge_length, base)
    
    ls_refractive_index = lm.ls_refractive_index
    lensmat_refractive_index = lm.lensmat_refractive_index
    diameter = 2*max_radius*diameter_ratio
    thickness = diameter*thickness_ratio
    x = diameter/4
    a = 2*thickness/diameter**2

    H = a/4*diameter**2
    u = np.arctan(2*a*x)
    m = -np.tan(np.pi/2-u+np.arcsin(ls_refractive_index/lensmat_refractive_index*np.sin(u)))
    b = a*x**2-m*x-H
    X = (m+np.sqrt(m**2+4*a*(-a*x**2+m*x+2*H)))/(-2*a)
    p = np.arctan((2*a*m*X-1)/(2*a*X+m))
    q = np.arcsin(lensmat_refractive_index/ls_refractive_index*np.sin(p))
    M = (1+2*a*X*np.tan(q))/(2*a*X-np.tan(q))
    focal_length = -a*X**2+H-M*X

    side_length = 2*np.sqrt(3)/phi*focal_length + phi*edge_length

    def triangle_mesh(side_length, height):
        return make.linear_extrude([0, -side_length/2.0, side_length/2.0], [side_length/np.sqrt(3), -np.sqrt(3)/6*side_length, -np.sqrt(3)/6*side_length], thickness)

    for k in range(20):
        kabamland.add_solid(Solid(triangle_mesh(side_length, thickness), lm.lensmat, lm.ls, black_surface, 0xffff00), rotation=np.dot(make_rotation_matrix(spin_angle[k], direction[k]), make_rotation_matrix(angle[k], axis[k])), displacement=focal_length*normalize(facecoords[k])+facecoords[k]+thickness/2.0)


if __name__ == '__main__':
    from chroma import sample
    from chroma.generator import vertex
    import matplotlib.pyplot as plt

    kabamland  = Detector(lm.ls)
    build_lens_icosahedron(10, 3, .75, 0.1)
    build_pmt_icosahedron(10, 3, np.sqrt(3)/2.0, 0.25)
    view(kabamland)
    

