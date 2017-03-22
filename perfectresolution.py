from chroma import make, view
from chroma.geometry import Geometry, Material, Mesh, Solid, Surface
from chroma.demo.optics import glass, black_surface
from chroma.detector import Detector
from chroma.pmt import build_pmt
from chroma.sample import uniform_sphere
from chroma.transform import make_rotation_matrix, normalize
import detectorconfig
import meshhelper as mh
import lensmaterials as lm
import numpy as np

inputn = 16.0
def lens(diameter, thickness, nsteps=inputn):
    #constructs a parabolic lens
    a = np.linspace(0, diameter/2, nsteps/2, endpoint=False)
    b = np.linspace(diameter/2, 0, nsteps/2)
    return make.rotate_extrude(np.concatenate((a, b)), np.concatenate((2*thickness/diameter**2*(a)**2-0.5*thickness, -2.0*thickness/diameter**2*(b)**2+0.5*thickness)), nsteps=inputn)

##new
## note that I should not use -ys (makes the mesh tracing clockwise)
def pclens2(radius, diameter, nsteps=inputn):
    #works best with angles endpoint=True
    halfd = diameter/2.0
    shift = np.sqrt(radius**2-(halfd)**2)
    theta = np.arctan(shift/(halfd))
    angles = np.linspace(theta, np.pi/2, nsteps)
    x = radius*np.cos(angles)
    y = radius*np.sin(angles) - shift
    xs = np.concatenate((np.zeros(1), x))
    ys = np.concatenate((np.zeros(1), y))
    return make.rotate_extrude(xs, ys, nsteps=inputn)

def spherical_lens(R1, R2, diameter, nsteps=16):
    '''constructs a spherical lens with specified radii of curvature. Works with meniscus lenses. Make sure not to fold R1 through R2 or vica-versa in order to keep rotate_extrude going counterclockwise.
    shift is the amount needed to move the hemisphere in the y direction to make the spherical cap. 
    R1 goes towards positive y, R2 towards negative y.'''
    if (abs(R1) < diameter/2.0) or (abs(R2) < diameter/2.0):
        raise Exception('R1 and R2 must be larger than diameter/2.0')
    signR1 = np.sign(R1)
    signR2 = np.sign(R2)
    shift1 = -signR1*np.sqrt(R1**2 - (diameter/2.0)**2)
    shift2 = -signR2*np.sqrt(R2**2 - (diameter/2.0)**2)
    theta1 = np.arctan(-shift1/(diameter/2.0))
    theta2 = np.arctan(-shift2/(diameter/2.0))
    angles1 = np.linspace(theta1, signR1*np.pi/2, nsteps/2)
    angles2 = np.linspace(signR2*np.pi/2, theta2, nsteps/2, endpoint=False)
    x1 = abs(R1*np.cos(angles1))
    x2 = abs(R2*np.cos(angles2))
    y1 = signR1*R1*np.sin(angles1) + shift1
    y2 = signR2*R2*np.sin(angles2) + shift2
    # thickness = y1[nsteps/2-1]-y2[0]
    # print 'thickness: ' + str(thickness)
    return make.rotate_extrude(np.concatenate((x2,x1)), np.concatenate((y2,y1)), nsteps=16)

def disk(radius, nsteps=inputn):
    return make.rotate_extrude([0, radius], [0, 0], nsteps)
 
##end new

def cylindrical_shell(inner_radius, outer_radius, thickness, nsteps=inputn):
    #make sure that nsteps is the same as that of rotate extrude in lens
    #inner_radius must be less than outer_radius
    return make.rotate_extrude([inner_radius, outer_radius, outer_radius, inner_radius], [-thickness/2.0, -thickness/2.0, thickness/2.0, thickness/2.0], nsteps)

def inner_blocker_mesh(radius, thickness, nsteps=inputn):
    #creates a mesh of the curved triangular shape between three tangent congruent circles.
    right_angles = np.linspace(np.pi, 2*np.pi/3, nsteps, endpoint=False)
    top_angles = np.linspace(5*np.pi/3, 4*np.pi/3, nsteps, endpoint=False)
    left_angles = np.linspace(np.pi/3, 0, nsteps, endpoint=False)
    rightx = radius*np.cos(right_angles) + radius
    righty = radius*np.sin(right_angles) - np.sqrt(3)/3.0*radius
    topx = radius*np.cos(top_angles)
    topy = radius*np.sin(top_angles) + 2*radius/np.sqrt(3)
    leftx = radius*np.cos(left_angles) - radius
    lefty = radius*np.sin(left_angles) - np.sqrt(3)*radius/3.0
    xs = np.concatenate((rightx, topx, leftx))
    ys = np.concatenate((righty, topy, lefty))
    return make.linear_extrude(xs, ys, thickness)
                  
def outer_blocker_mesh(radius, thickness, nsteps=16):
    #produces half of the shape that is between four circles in a square array. 
    #the center is halfway along the flat side.
    right_angles = np.linspace(3*np.pi/2.0, np.pi, nsteps)
    left_angles = np.linspace(0, -np.pi/2.0, nsteps)
    rightx = radius*np.cos(right_angles) + radius
    righty = radius*np.sin(right_angles) + radius
    leftx = radius*np.cos(left_angles) - radius
    lefty = radius*np.sin(left_angles) + radius
    xs = np.concatenate((rightx, leftx))
    ys = np.concatenate((righty, lefty))
    return make.linear_extrude(xs, ys, thickness) 

def corner_blocker_mesh(radius, thickness, nsteps=inputn):
    #constructs triangular corners with a single curved side.
    #center is at the point connecting the two straight edges.
    angles = np.linspace(5*np.pi/6.0, np.pi/6.0, nsteps)
    bottomx = radius*np.cos(angles)
    bottomy = radius*np.sin(angles)-2*radius
    xs = np.append(bottomx, 0)
    ys = np.append(bottomy, 0)
    return make.linear_extrude(xs, ys, thickness)   
                         
def triangle_mesh(side_length, thickness):
    return make.linear_extrude([0, -side_length/2.0, side_length/2.0], [side_length/np.sqrt(3), -np.sqrt(3)/6*side_length, -np.sqrt(3)/6*side_length], thickness)

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

def gaussian_sphere(pos, sigma, n):
    points = np.empty((n, 3))
    points[:,0] = np.random.normal(0.0, sigma, n) + pos[0]
    points[:,1] = np.random.normal(0.0, sigma, n) + pos[1]
    points[:,2] = np.random.normal(0.0, sigma, n) + pos[2]
    pos = points
    dir = uniform_sphere(n)
    pol = np.cross(dir, uniform_sphere(n))
    #300 nm is roughly the pseudocumene scintillation wavelength
    wavelengths = np.repeat(300.0, n)
    return Photons(pos, dir, pol, wavelengths) 

def uniform_photons(edge_length, n):
    #constructs photons uniformly throughout the detector inside of the inscribed sphere.
    inscribed_radius = np.sqrt(3)/12*(3+np.sqrt(5))*edge_length
    radius_root = inscribed_radius*np.random.uniform(0.0, 1.0, n)**(1.0/3)
    theta = np.arccos(np.random.uniform(-1.0, 1.0, n))
    phi = np.random.uniform(0.0, 2*np.pi, n)
    points = np.empty((n,3))
    points[:,0] = radius_root*np.sin(theta)*np.cos(phi)
    points[:,1] = radius_root*np.sin(theta)*np.sin(phi)
    points[:,2] = radius_root*np.cos(theta)
    pos = points
    dir = uniform_sphere(n)
    pol = np.cross(dir, uniform_sphere(n))
    #300 nm is roughly the pseudocumene scintillation wavelength
    wavelengths = np.repeat(300.0, n)
    return Photons(pos, dir, pol, wavelengths) 

def find_max_radius(edge_length, base):
    #finds the maximum possible radius for the lenses on a face.
    max_radius = edge_length/(2*(base+np.sqrt(3)-1))
    return max_radius

def find_inscribed_radius(edge_length):
    #finds the inscribed radius of the lens_icoshadron
    inscribed_radius = np.sqrt(3)/12.0*(3+np.sqrt(5))*edge_length
    return inscribed_radius

def triangular_indices(base):
    # produces the x and y indices for a triangular array of points, given the amount of points at the base layer.
    xindices = np.linspace(0, 2*(base-1), base)
    yindices = np.repeat(0, base)
    for i in np.linspace(1, base-1, base-1):
        xindices = np.append(xindices, np.linspace(i, 2*(base-1)-i, base-i))
        yindices = np.append(yindices, np.repeat(i, base-i))
    return xindices, yindices

def triangular_number(base):
    return base*(base+1)/2

def find_packing_ratio(base, diameter_ratio=1.0):
    #approaches np.pi/(2*np.sqrt(3)) = 0.90689968
    radius = find_max_radius(1.0, base)*diameter_ratio
    lens_area = triangular_number(base)*np.pi*radius**2
    side_area = np.sqrt(3)/4.0
    packing_ratio = lens_area/side_area
    return packing_ratio

print find_packing_ratio(9)
                
def return_values(edge_length, base):
    edge_length = float(edge_length)
    phi = (1+np.sqrt(5))/2.0

    #lists of the coordinate centers of each face and vertices of the icosahedron
    facecoords = np.array([[phi**2/6*edge_length, phi**2/6*edge_length, phi**2/6*edge_length], [phi**2/6*edge_length, phi**2/6*edge_length, -phi**2/6*edge_length], [phi**2/6*edge_length, -phi**2/6*edge_length, phi**2/6*edge_length], [phi**2/6*edge_length, -phi**2/6*edge_length, -phi**2/6*edge_length], [-phi**2/6*edge_length, phi**2/6*edge_length, phi**2/6*edge_length], [-phi**2/6*edge_length, phi**2/6*edge_length, -phi**2/6*edge_length], [-phi**2/6*edge_length, -phi**2/6*edge_length, phi**2/6*edge_length], [-phi**2/6*edge_length, -phi**2/6*edge_length, -phi**2/6*edge_length], [0.0, edge_length*phi/6, edge_length*(2*phi+1)/6], [0.0, edge_length*phi/6, -edge_length*(2*phi+1)/6], [0.0, -edge_length*phi/6, edge_length*(2*phi+1)/6], [0.0, -edge_length*phi/6, -edge_length*(2*phi+1)/6], [edge_length*phi/6, edge_length*(2*phi+1)/6, 0.0], [edge_length*phi/6, -edge_length*(2*phi+1)/6, 0.0], [-edge_length*phi/6, edge_length*(2*phi+1)/6, 0.0], [-edge_length*phi/6, -edge_length*(2*phi+1)/6, 0.0], [edge_length*(2*phi+1)/6, 0.0, edge_length*phi/6], [edge_length*(2*phi+1)/6, 0.0, -edge_length*phi/6], [-edge_length*(2*phi+1)/6, 0.0, edge_length*phi/6], [-edge_length*(2*phi+1)/6, 0.0, -edge_length*phi/6]])

    vertices = np.array([[edge_length*phi/2, edge_length/2, 0], [edge_length*phi/2, edge_length/2, 0], [edge_length*phi/2, -edge_length/2, 0], [edge_length*phi/2, -edge_length/2, 0], [-edge_length*phi/2, edge_length/2, 0], [-edge_length*phi/2, edge_length/2, 0], [-edge_length*phi/2, -edge_length/2, 0], [-edge_length*phi/2, -edge_length/2, 0], [-edge_length/2, 0, edge_length*phi/2], [edge_length/2, 0, -edge_length*phi/2], [-edge_length/2, 0, edge_length*phi/2], [edge_length/2, 0, -edge_length*phi/2], [edge_length*phi/2, edge_length/2, 0], [edge_length*phi/2, -edge_length/2, 0], [-edge_length*phi/2, edge_length/2, 0], [-edge_length*phi/2, -edge_length/2, 0], [edge_length*phi/2, edge_length/2, 0], [edge_length*phi/2, -edge_length/2, 0], [-edge_length*phi/2, edge_length/2, 0], [-edge_length*phi/2, -edge_length/2, 0]])

    #rotating each face onto the plane orthogonal to a line from the origin to the center of the face.
    direction = -normalize(facecoords)
    axis = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    angle = np.arccos(direction[:,2])
    
    #spinning each face into its correct orientation within the plane that is orthogonal to a line from the origin to the center of the face.
    A = np.empty((20, 3))
    B = np.empty((20, 3))
    spin_sign = np.empty(20)
    spin_angle = np.empty(20)
    for k in range(20):
        A[k] = np.dot(make_rotation_matrix(angle[k], axis[k]), np.array([0, edge_length/np.sqrt(3), 0]))
        B[k] = vertices[k] - facecoords[k]
        spin_sign[k] = np.sign(np.dot(np.dot(A[k], make_rotation_matrix(np.pi/2, facecoords[k])), B[k]))
        spin_angle[k] = spin_sign[k]*np.arccos(3*np.dot(A[k], B[k])/edge_length**2)
        
    return edge_length, facecoords, direction, axis, angle, spin_angle
  
def build_lens_icosahedron(edge_length, base, diameter_ratio, thickness_ratio, blockers=True, blocker_thickness_ratio=1.0/1000):
    """input edge length of icosahedron 'edge_length', the number of small triangles in the base of each face 'base', the ratio of the diameter of each lens to the maximum diameter possible 'diameter_ratio', the ratio of the thickness of the lens to the chosen (not maximum) diameter 'thickness_ratio', and the ratio of the thickness of the blockers to that of the lenses 'blocker_thickness_ratio' to return the icosahedron of lenses in kabamland.
    """
    edge_length, facecoords, direction, axis, angle, spin_angle = return_values(edge_length, base)
    max_radius = find_max_radius(edge_length, base)
    xshift = edge_length/2.0
    yshift = edge_length/(2.0*np.sqrt(3))

    #iterating the lenses into a hexagonal pattern within a single side using triangular numbers. First, coordinate indices are created, and then these are transformed into the actual coordinate positions based on the parameters given.
    lens_xindices, lens_yindices = triangular_indices(base)
    first_lens_xcoord = np.sqrt(3)*max_radius
    first_lens_ycoord = max_radius
    lens_xcoords = max_radius*lens_xindices + first_lens_xcoord - xshift
    lens_ycoords = np.sqrt(3)*max_radius*lens_yindices + first_lens_ycoord - yshift

    #creating the lenses for a single face
    ##changed
    #I am using pclens1 and changed the rotation matrix to try and keep the curved surface towards the interior
    lensdiameter = 2*diameter_ratio*max_radius
    print 'lensdiameter: ' + str(lensdiameter)
    pcrad = 0.9*lensdiameter
    R1 = 0.584
    R2 = -9.151
    #initial_lens = mh.rotate(spherical_lens(R1, R2, lensdiameter), make_rotation_matrix(-np.pi/2, (1,0,0)))
    initial_lens = mh.rotate(pclens2(pcrad, lensdiameter), make_rotation_matrix(-np.pi/2, (1,0,0)))
    #initial_lens = mh.rotate(disk(lensdiameter/2.0), make_rotation_matrix(-np.pi/2, (1,0,0)))
    ##end changed

    face = Solid(mh.shift(initial_lens, (lens_xcoords[0], lens_ycoords[0], 0)), lm.lensmat, lm.ls)  
    for i in np.linspace(1, triangular_number(base)-1, triangular_number(base)-1):
        face = face + Solid(mh.shift(initial_lens, (lens_xcoords[i], lens_ycoords[i], 0)), lm.lensmat, lm.ls) 
    
    #creating the various blocker shapes to fill in the empty space of a single face.
    if blockers:
        blocker_thickness = 2*max_radius*diameter_ratio*thickness_ratio*blocker_thickness_ratio

        if base >= 2:
            down_blocker = inner_blocker_mesh(max_radius, blocker_thickness)
            down_blocker_xindices, down_blocker_yindices = triangular_indices(base-1)
            first_down_blocker_xcoord = max_radius*(np.sqrt(3) + 1)
            first_down_blocker_ycoord = max_radius*(np.sqrt(3)/3.0 + 1)
            down_blocker_xcoords = max_radius*down_blocker_xindices + first_down_blocker_xcoord - xshift
            down_blocker_ycoords = np.sqrt(3)*max_radius*down_blocker_yindices + first_down_blocker_ycoord - yshift
            for i in range(triangular_number(base-1)):
                face = face + Solid(mh.shift(down_blocker, (down_blocker_xcoords[i], down_blocker_ycoords[i], 0)), lm.lensmat, lm.ls, black_surface, 0xff0000)

            bottom_blocker = outer_blocker_mesh(max_radius, blocker_thickness)
            right_blocker = mh.rotate(bottom_blocker, make_rotation_matrix(-2*np.pi/3.0, (0, 0, 1)))
            left_blocker = mh.rotate(bottom_blocker, make_rotation_matrix(2*np.pi/3.0, (0, 0, 1)))
            distances = max_radius*(2*np.arange(base-1) + np.sqrt(3) + 1)
            bottom_blocker_xcoords = distances - xshift
            right_blocker_xcoords = distances*np.cos(2*np.pi/3.0) + xshift
            right_blocker_ycoords = distances*np.sin(2*np.pi/3.0) - yshift
            left_blocker_xcoords = distances*np.cos(np.pi/3.0) - xshift
            left_blocker_ycoords = distances*np.sin(np.pi/3.0) - yshift
            for i in range(base-1):
                face = face + Solid(mh.shift(bottom_blocker, (bottom_blocker_xcoords[i], -yshift, 0)), lm.lensmat, lm.ls, black_surface, 0xff0000)
                face = face + Solid(mh.shift(right_blocker, (right_blocker_xcoords[i], right_blocker_ycoords[i], 0)), lm.lensmat, lm.ls, black_surface, 0xff0000)
                face = face + Solid(mh.shift(left_blocker, (left_blocker_xcoords[i], left_blocker_ycoords[i], 0)), lm.lensmat, lm.ls, black_surface, 0xff0000)

        if base >= 3:
            up_blocker = mh.rotate(down_blocker, make_rotation_matrix(np.pi, (0, 0, 1)))
            up_blocker_xindices, up_blocker_yindices = triangular_indices(base-2)
            first_up_blocker_xcoord = max_radius*(np.sqrt(3) + 2)
            first_up_blocker_ycoord = max_radius*(2.0/np.sqrt(3) + 1)
            up_blocker_xcoords = max_radius*up_blocker_xindices + first_up_blocker_xcoord - xshift
            up_blocker_ycoords = np.sqrt(3)*max_radius*up_blocker_yindices + first_up_blocker_ycoord - yshift
            for i in range(triangular_number(base-2)):    
                face = face + Solid(mh.shift(up_blocker, (up_blocker_xcoords[i], up_blocker_ycoords[i], 0)), lm.lensmat, lm.ls, black_surface, 0xff0000)
        
        corner_blocker = corner_blocker_mesh(max_radius, blocker_thickness)
        for i in range(3):
            theta = 2*np.pi/3.0*i + np.pi/2.0 
            rotated_corner_blocker = mh.rotate(corner_blocker, make_rotation_matrix(-2*np.pi/3.0*i, (0, 0, 1)))
            face = face + Solid(mh.shift(rotated_corner_blocker, (2*yshift*np.cos(theta), 2*yshift*np.sin(theta), 0)), lm.lensmat, lm.ls, black_surface, 0xff0000)

        if diameter_ratio < 1:
            annulus_blocker = mh.rotate(cylindrical_shell(diameter_ratio*max_radius, max_radius, blocker_thickness), make_rotation_matrix(np.pi/2.0, (1,0,0)))
            for i in range(triangular_number(base)):
                face = face + Solid(mh.shift(annulus_blocker, (lens_xcoords[i], lens_ycoords[i], 0)), lm.lensmat, lm.ls, black_surface, 0xff0000)

    #creating all 20 faces and putting them into the detector with the correct orientations.
    for k in range(20):   
        kabamland.add_solid(face, rotation=np.dot(make_rotation_matrix(spin_angle[k], direction[k]), make_rotation_matrix(angle[k], axis[k])), displacement=facecoords[k])

def find_thickness_ratio(edge_length, base, diameter_ratio, focal_length):
    #finds the thickness ratio to make a given lens have a particular focal length
    def F(t_r):
        return (find_focal_length(edge_length, base, diameter_ratio, t_r) - focal_length)**2
    thickness_ratio = optimize.fmin(F, 0.001, xtol=1e-6)
    #starting off with a low initial guess takes longer but gets the correct answer more of the time.
    return thickness_ratio

def build_pmt_icosahedron(edge_length, base, diameter_ratio, thickness_ratio):
    edge_length, facecoords, direction, axis, angle, spin_angle = return_values(edge_length, base)
    ##changed
    #focal_length = find_focal_length(edge_length, base, diameter_ratio, thickness_ratio)
    max_radius = find_max_radius(edge_length, base)
    lensdiameter = 2*diameter_ratio*max_radius
    focal_length = 1.00
    ##end changed
    #creation of triangular pmts arranged around the inner icosahedron
    pmt_side_length = np.sqrt(3)*(3-np.sqrt(5))*focal_length + edge_length
    for k in range(20):
        kabamland.add_pmt(Solid(triangle_mesh(pmt_side_length, 0.000001), glass, lm.ls, lm.fulldetect, 0xBBFFFFFF), rotation=np.dot(make_rotation_matrix(spin_angle[k], direction[k]), make_rotation_matrix(angle[k], axis[k])), displacement=facecoords[k] + focal_length*normalize(facecoords[k]) + 0.0000005*normalize(facecoords[k]))

def build_kabamland(configname):
    config = detectorconfig.configdict[configname]
    #build_lens_icosahedron(config.edge_length, config.base, config.diameter_ratio, config.thickness_ratio, config.blockers, config.blocker_thickness_ratio)
    build_pmt_icosahedron(config.edge_length, config.base, config.diameter_ratio, config.thickness_ratio)

if __name__ == '__main__':
    from chroma.sim import Simulation
    from chroma import sample
    from chroma.event import Photons
    from chroma.loader import load_bvh
    from chroma.generator import vertex
    from ShortIO.root_short import ShortRootWriter
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    kabamland = Detector(lm.ls)
    datadir = "/home/miladmalek/TestData/"

    def create_event(location, sigma, amount, config, eventname):
        #simulates a single event within the detector for a given configuration.
        build_kabamland(config)
        kabamland.flatten()
        kabamland.bvh = load_bvh(kabamland)
        f = ShortRootWriter(datadir + eventname)
        sim = Simulation(kabamland)
        sim_event = gaussian_sphere(location, sigma, amount)
        for ev in sim.simulate(sim_event, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
            f.write_event(ev)
        f.close()

    def create_double_source_event(loc1, loc2, sigma, amount, config, eventname):
        #simulates an event with two different photon sources for a given configuration.
        build_kabamland(config)
        kabamland.flatten()
        kabamland.bvh = load_bvh(kabamland)
        f = ShortRootWriter(datadir + eventname)
        sim = Simulation(kabamland)
        sim_event1 = gaussian_sphere(loc1, sigma, amount)
        sim_event2 = gaussian_sphere(loc2, sigma, amount)
        sim_event = sim_event1+sim_event2 # Just add the list of photons from the two sources into a single event
        for ev in sim.simulate(sim_event, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
            f.write_event(ev)
        f.close()

    def full_detector_simulation(edge_length, amount, config, simname):
        #simulates 1000*amount photons uniformly spread throughout a sphere whose radius is the inscribed radius of the icosahedron. Note that viewing may crash if there are too many lenses. (try using configview)
        build_kabamland(config)
        kabamland.flatten()
        kabamland.bvh = load_bvh(kabamland)
        # view(kabamland)
        # quit()
        f = ShortRootWriter(datadir + simname)
        sim = Simulation(kabamland)
        for j in range(100):
            print j
            sim_events = [uniform_photons(edge_length, amount) for i in range(10)]
            for ev in sim.simulate(sim_events, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
                f.write_event(ev)
        f.close()


    create_event((3,3,3), 0.1, 100000, 'configpc7', 'event-perfectres-configpc7-f_l-1-(3-3-3)-01-100000.root')
    #full_detector_simulation(10.0, 10, 'configpc7', 'sim-configpc7-f_l-1-perfectres-10000.root')
        
    
    #create_event((3,3,3), 0.1, 100000, 'configpc7', 'event-configpc7-f_l-1-(3-3-3)-01-100000.root')
    
    #create_double_source_event((0,0,0), (3,3,3), 0.1, 100, 'configpc7', 'event-configpc7-f_l-1-(0-0-0)-(3-3-3)-01-100.root')
 
    #full_detector_simulation(10.0, 1000, 'configpc2', 'sim-configpc2-million.root') 
        
    #create_event((2,2,2), 0.1, 100000, 'configpc7', 'event-configpc7-meniscus2-(2-2-2)-01-100000.root')
    
    #create_event((3,3,3), 0.1, 100000, 'configpc7', 'event-configpc7-meniscus2-(3-3-3)-01-100000.root')
    
    #create_event((6,6,6), 0.1, 100000, 'configpc7', 'event-configpc7-meniscus2-(6-6-6)-01-100000.root')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # xs = ev.photons_end.pos[detected][:,0]
    # ys = ev.photons_end.pos[detected][:,1]
    # zs = ev.photons_end.pos[detected][:,2]
    # ax.scatter(xs, ys, zs)
    # ax.set_xlabel('X final position')
    # ax.set_ylabel('Y final position')
    # ax.set_zlabel('Z final position')
    # plt.show()
