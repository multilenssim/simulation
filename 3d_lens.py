from chroma import make, view, sample, gpu
from chroma.geometry import Geometry, Material, Mesh, Solid, Surface
from chroma.demo.optics import glass, black_surface
from chroma.detector import Detector
from chroma.sim import Simulation
from chroma.generator import vertex
from chroma.pmt import build_pmt
from chroma.sample import uniform_sphere
from chroma.transform import make_rotation_matrix, normalize
from chroma.event import Photons
from chroma.loader import load_bvh
from chroma.generator import vertex
import chroma.make as mk
from ShortIO.root_short import ShortRootWriter
import detectorconfig
import lenssystem
import meshhelper as mh
import lensmaterials as lm
import numpy as np
from scipy.spatial import distance
from matplotlib.tri import Triangulation
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    #creates an equilateral triangle centered at its centroid.
    return make.linear_extrude([0, -side_length/2.0, side_length/2.0], [side_length/np.sqrt(3), -np.sqrt(3)/6*side_length, -np.sqrt(3)/6*side_length], thickness)



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
  	

def curved_surface(detector_r=1.0, diameter = 2.5, nsteps=10):
    '''Builds a curved surface based on the specified radius. Origin is center of surface.'''
    if (detector_r < diameter/2.0):
        raise Exception('The Radius of the curved surface must be larger than diameter/2.0')
    
    shift1 = -np.sqrt(detector_r**2 - (diameter/2.0)**2)
    theta1 = np.arctan(-shift1/(diameter/2.0))
    angles1 = np.linspace(theta1, np.pi/2, nsteps/2.0)
    x_value = abs(detector_r*np.cos(angles1))
    y_value = detector_r*np.sin(angles1) - detector_r

    return  make.rotate_extrude(x_value, y_value, nsteps)    
        
def build_single_lens(kabamland, edge_length, base, diameter_ratio, thickness_ratio, half_EPD, blockers=True, blocker_thickness_ratio=1.0/1000, light_confinement=False, focal_length=1.0, lens_system_name=None):
    """input edge length of icosahedron 'edge_length', the number of small triangles in the base of each face 'base', the ratio of the diameter of each lens to the maximum diameter possible 'diameter_ratio' (or the fraction of the default such ratio, if a curved detector lens system), the ratio of the thickness of the lens to the chosen (not maximum) diameter 'thickness_ratio', the radius of the blocking entrance pupil 'half_EPD', and the ratio of the thickness of the blockers to that of the lenses 'blocker_thickness_ratio' to return the icosahedron of lenses in kabamland. Light_confinment=True adds cylindrical shells behind each lens that absorb all the light that touches them, so that light doesn't overlap between lenses. If lens_system_name is a string that matches one of the lens systems in lenssystem.py, the corresponding lenses and detectors will be built. Otherwise, a default simple lens will be built, with parameters hard-coded below.
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
    if not lens_system_name in lenssystem.lensdict: # Lens system isn't recognized
        print 'Warning: lens system name '+str(lens_system_name)+' not recognized; using default lens.'    ##changed
        #I changed the rotation matrix to try and keep the curved surface towards the interior
        #focal_length = 1.0
        lensdiameter = 2*diameter_ratio*max_radius
        #print 'lensdiameter: ' + str(lensdiameter)
        pcrad = 0.9*lensdiameter 
        R1 = 0.584*lensdiameter # meniscus 6 values
        R2 = -9.151*lensdiameter

        #as_solid = Solid(as_mesh, lm.lensmat, lm.ls)    
        initial_lens = as_mesh

        initial_lens = mh.rotate(spherical_lens(R1, R2, lensdiameter), make_rotation_matrix(-np.pi/2, (1,0,0))) # meniscus 6 lens
        
        #initial_lens = mh.rotate(pclens2(pcrad, lensdiameter), make_rotation_matrix(-np.pi/2, (1,0,0)))
        #initial_lens = mh.rotate(disk(lensdiameter/2.0), make_rotation_matrix(-np.pi/2, (1,0,0)))
        ##end changed
        lenses = [initial_lens]
    else: # Get the list of lens meshes from the appropriate lens system
        scale_rad = max_radius*diameter_ratio
        lenses = lenssystem.get_lens_mesh_list(lens_system_name, scale_rad)
    
    i=k=0
    lenses[0] = mh.shift(lenses[0], (lens_xcoords[np.int(i)], lens_ycoords[np.int(i)], 0.))
    lenses2 = lenses[0]
    lenses[0] = mh.rotate(lenses[0], np.dot(make_rotation_matrix(spin_angle[k], direction[k]), make_rotation_matrix(angle[k], axis[k])))
    lenses[0] = mh.shift(lenses[0], facecoords[k])
    

    face = Solid(lenses[0], lm.lensmat, lm.ls) 
    #kabamland.add_solid(face, rotation=np.dot(make_rotation_matrix(spin_angle[k], direction[k]), make_rotation_matrix(angle[k], axis[k])), displacement=facecoords[k])
    return lenses[0], lenses2
	       
def build_single_curvedsurface(kabamland, edge_length, base, diameter_ratio, focal_length=1.0, detector_r = 1.0, nsteps = 10):
    
    edge_length, facecoords, direction, axis, angle, spin_angle = return_values(edge_length, base)
    max_radius = find_max_radius(edge_length, base)
    diameter = max_radius*2.0
    xshift = edge_length/2.0
    yshift = edge_length/(2.0*np.sqrt(3))

    #iterating the lenses into a hexagonal pattern within a single side using triangular numbers. First, coordinate indices are created, and then these are transformed into the actual coordinate positions based on the parameters given.
    lens_xindices, lens_yindices = triangular_indices(base)
    first_lens_xcoord = np.sqrt(3)*max_radius
    first_lens_ycoord = max_radius
    lens_xcoords = max_radius*lens_xindices + first_lens_xcoord - xshift
    lens_ycoords = np.sqrt(3)*max_radius*lens_yindices + first_lens_ycoord - yshift
    
    #Changed the rotation matrix to try and keep the curved surface towards the interior
    initial_curved_surf = mh.rotate(curved_surface(detector_r, diameter=diameter, nsteps=nsteps), make_rotation_matrix(+np.pi/2, (1,0,0)))
    initial_curved_surf2 = initial_curved_surf
    initial_curved_surf = mh.shift(initial_curved_surf, (lens_xcoords[0], lens_ycoords[0], 0))
    k=0 
    initial_curved_surf = mh.rotate(initial_curved_surf, np.dot(make_rotation_matrix(spin_angle[k], direction[k]), make_rotation_matrix(angle[k], axis[k])))
    initial_curved_surf = mh.shift(initial_curved_surf, facecoords[k] + focal_length*normalize(facecoords[k]))
    

    face = Solid(initial_curved_surf, lm.ls, lm.ls, lm.fulldetect, 0x0000FF)   
    #kabamland.add_solid(face, rotation=np.dot(make_rotation_matrix(spin_angle[k], direction[k]), make_rotation_matrix(angle[k], axis[k])), displacement=facecoords[k] + focal_length*normalize(facecoords[k]))
    return initial_curved_surf, initial_curved_surf2

def build_pmt_icosahedron(kabamland, edge_length, base, focal_length=1.0):
    edge_length, facecoords, direction, axis, angle, spin_angle = return_values(edge_length, base)
    pmt_side_length = np.sqrt(3)*(3-np.sqrt(5))*focal_length + edge_length
    #for k in (0,1,2,3,4,5,6,7,9,11,12,13,14,15,16,17,18,19):
    for k in range(20):
        kabamland.add_pmt(Solid(triangle_mesh(pmt_side_length, .001*pmt_side_length), glass, lm.ls, lm.fullabsorb, 0xBBFFFFFF), rotation=np.dot(make_rotation_matrix(spin_angle[k], direction[k]), make_rotation_matrix(angle[k], axis[k])), displacement=facecoords[k] + focal_length*normalize(facecoords[k]) + 0.0000005*normalize(facecoords[k]))

def build_kabamland(kabamland, configname):
    # focal_length sets dist between lens plane and PMT plane (or back of curved detecting surface);
    #(need not equal true lens focal length)
    config = detectorconfig.configdict[configname]

    #build_pmt_icosahedron(kabamland, config.edge_length, config.base, focal_length=config.focal_length*1.5) # Built further out, just as a way of stopping photons
    
    surf, surf2 = build_single_curvedsurface(kabamland, config.edge_length, config.base, config.diameter_ratio, focal_length=config.focal_length, detector_r = config.detector_r, nsteps = config.nsteps)
    
    lens, lens2 = build_single_lens(kabamland, config.edge_length, config.base, config.diameter_ratio, config.thickness_ratio, config.half_EPD, config.blockers, blocker_thickness_ratio=config.blocker_thickness_ratio, light_confinement=config.light_confinement, focal_length=config.focal_length, lens_system_name=config.lens_system_name)
	
    a = (np.mean(lens.get_triangle_centers()[:,0]), np.mean(lens.get_triangle_centers()[:,1]), np.mean(lens.get_triangle_centers()[:,2]))
    b =  (np.mean(surf.get_triangle_centers()[:,0]), np.mean(surf.get_triangle_centers()[:,1]), np.mean(surf.get_triangle_centers()[:,2]))
    
    a2 = (np.mean(lens2.get_triangle_centers()[:,0]), np.mean(lens2.get_triangle_centers()[:,1]), np.mean(lens2.get_triangle_centers()[:,2]))
    b2 =  (np.mean(surf2.get_triangle_centers()[:,0]), np.mean(surf2.get_triangle_centers()[:,1]), np.mean(surf2.get_triangle_centers()[:,2]))
    surf2 = mh.shift(surf2, (b2[0], b2[1],0))
    lens2 = mh.shift(lens2, (-a2[0], -a2[1],0))
    a2 = (np.mean(lens2.get_triangle_centers()[:,0]), np.mean(lens2.get_triangle_centers()[:,1]), np.mean(lens2.get_triangle_centers()[:,2]))
    b2 =  (np.mean(surf2.get_triangle_centers()[:,0]), np.mean(surf2.get_triangle_centers()[:,1]), np.mean(surf2.get_triangle_centers()[:,2]))
    lens2 = mh.shift(lens2, (-a2[0], -a2[1],(distance.euclidean(a2,b2)+distance.euclidean(a,b))))
    a2 = (np.mean(lens2.get_triangle_centers()[:,0]), np.mean(lens2.get_triangle_centers()[:,1]), np.mean(lens2.get_triangle_centers()[:,2]))
    b2 =  (np.mean(surf2.get_triangle_centers()[:,0]), np.mean(surf2.get_triangle_centers()[:,1]), np.mean(surf2.get_triangle_centers()[:,2]))

    kabamland.add_solid(Solid(surf2, lm.ls, lm.ls, lm.fulldetect, 0x0000FF), rotation=None, displacement=(0,0,0))
    kabamland.add_solid(Solid(lens2, lm.lensmat, lm.ls) , rotation=None, displacement=None)
    
    pos = 5
    off = (pos-1)*100/2
    upshift = 500
    for i in range(pos):
		for k in range(pos):
			if(k==0 and i==0):
				connect = mk.box(60, 60, 60, center=(a2[0]+off, a2[1]+off, a2[2]+upshift))
			else: 
				connect = connect + mk.box(60, 60, 60, center=(a2[0]-i*100+off, a2[1]-k*100+off, a2[2]+upshift))
    box = Solid(connect, lm.ls, lm.ls, lm.fulldetect, 0x0000FF)
    #kabamland.add_solid(box, rotation=None, displacement=(0,0,0))
    
    Photons,photon_pos = photon_angle(a2, pos)
    
    return Photons, lens2, surf2, photon_pos, pos
    
def getRandom():
	nthreads_per_block = 64
	max_blocks = 1024
	rng_states = gpu.get_rng_states(nthreads_per_block*max_blocks, seed=0)
	doRandom = [nthreads_per_block, max_blocks, rng_states]
	return doRandom  

def event_display(numPhotons, photon_track, vertex,surf2, lens2, pos, rep, config):
	
	fig = plt.figure(figsize=(20, 10))
	ax = fig.gca(projection='3d')
	
	ax._axis3don = False
	lens2.remove_duplicate_vertices()
	lens_vertices = lens2.assemble()
	surf_vertices = surf2.assemble()
	
	nr_triangles2 = len(surf2.get_triangle_centers()[:,1])
	
	X2 = lens_vertices[:,:,0].flatten()
	Y2 = lens_vertices[:,:,1].flatten()
	Z2 = lens_vertices[:,:,2].flatten()
	
	X3 = surf_vertices[:,:,0].flatten()
	Y3 = surf_vertices[:,:,1].flatten()
	Z3 = surf_vertices[:,:,2].flatten()

	triangles = [[3*ii,3*ii+1,3*ii+2] for ii in range(len(X2)/3)]
	triang = Triangulation(X2, Y2, triangles)
	
	triangles2 = [[3*ii,3*ii+1,3*ii+2] for ii in range(len(X3)/3)]
	triang2 = Triangulation(X3, Y3, triangles2)
	
	#ax.plot_trisurf(triang2, Z3, color="white", linewidth = 0.0, shade = True, alpha = 1.0)
	
	for ii in range(nr_triangles2/2):
		ax.plot(surf_vertices[ii,:,0], surf_vertices[ii,:,1], surf_vertices[ii,:,2], color='blue', lw = 1.)

	for i in range(numPhotons):
		X,Y,Z = photon_track[:,i,0].tolist(), photon_track[:,i,1].tolist(), photon_track[:,i,2].tolist()
		X.insert(0, pos[i,0])
		Y.insert(0, pos[i,1])
		Z.insert(0, pos[i,2])
		ax.plot(X,Y,Z, linewidth=2, color='r')	
		ax.scatter(X,Y,Z , c='k', s=0.)		
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax.view_init(elev=10, azim=90)
	
	ax.plot_trisurf(triang, Z2, color="white", shade = True, alpha = .90)
	
	plt.show()
		
def photon_angle(pos, rep):
    off = (rep-1)*100/2
    photon_pos = np.zeros((rep*rep,3))
    upshift = 800
    for i in range(rep):
		for k in range(rep):
			photon_pos[i*rep+k,:] = [pos[0]-i*100+off-100, pos[1]-k*100+off+400, pos[2]+upshift]
    photon_dir = np.tile(np.array([0.2,-0.55,-1]),(rep*rep,1))
    pol = np.cross(photon_dir, uniform_sphere(rep*rep))
    wavelength = np.repeat(300.0, rep*rep)
    return Photons(photon_pos, photon_dir, pol, wavelength), photon_pos


def propagate_photon(photon_type, numPhotons, nr_steps, geometry, nthreads_per_block, max_blocks, rng_states):
	gpu_photons = gpu.GPUPhotons(photon_type);
	gpu_geometry = gpu.GPUGeometry(geometry)
	photon_track = np.zeros((nr_steps, numPhotons, 3))
	for i in range(nr_steps):
		gpu_photons.propagate(gpu_geometry, rng_states, nthreads_per_block=nthreads_per_block, max_blocks=max_blocks, max_steps=1)
		photons = gpu_photons.get()
		photon_track[i,:,0] = photons.pos[:,0] 
		photon_track[i,:,1] = photons.pos[:,1] 
		photon_track[i,:,2] = photons.pos[:,2]
	return photons, photon_track


if __name__ == '__main__':

    datadir = "/home/miladmalek/TestData/"
    kabamland = Detector(lm.ls)
    config = "cfJiani3_2"
    events, lens2, surf2, pos, rep = build_kabamland(kabamland, config)
    kabamland.flatten()
    kabamland.bvh = load_bvh(kabamland)
    #view(kabamland)
    #quit()
    config2 = detectorconfig.configdict[config]
    sim = Simulation(kabamland, geant4_processes=0)
    runs = 1
    nr_hits = np.zeros((runs))
    doRandom = getRandom()
    numPhotons = rep*rep 
    for eg in range(runs):
		photons, photon_track = propagate_photon(events, numPhotons, 20, kabamland, doRandom[0], doRandom[1], doRandom[2])
		detected = (photons.flags & (0x1 << 2)).astype(bool) 
		nr_hits[eg] = len(photons.pos[detected])
		print "  Number of photons detected			", nr_hits[eg]
    
    vertex = photons.pos[detected] 
    event_display(numPhotons, photon_track, vertex, surf2, lens2, pos, rep, config2)
    
    
    
    #for ind, ev in enumerate(sim.simulate(events, keep_photons_beg=True,keep_photons_end=True, run_daq=True, max_steps=100)):
		#print ind 
		#detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)
		#transmit = (ev.photons_end.flags & (0x1 << 8)).astype(bool)
		#absorb = (ev.photons_end.flags & (0x1 << 3)).astype(bool) 
		
		
    
