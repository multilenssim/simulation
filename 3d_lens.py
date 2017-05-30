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

def find_max_radius(edge_length, base):
    #finds the maximum possible radius for the lenses on a face.
    max_radius = edge_length/(2*(base+np.sqrt(3)-1))
    return max_radius

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
    
    blocker = BuildBlocker(np.max(lens2.assemble()[:,:,0].flatten()), lens2.assemble()[:,:,2].flatten()[np.argmax(lens2.assemble()[:,:,0].flatten())])
    kabamland.add_solid(Solid(blocker, lm.ls, lm.ls, lm.fullabsorb, 0x0000FF), rotation=None, displacement=(0,0,0))
    
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
	
def BuildBlocker(radius, height): 
	nsteps = 30
	x_value = np.linspace(radius, radius+200, nsteps)
	y_value = [1]*nsteps
	
	blocker = make.rotate_extrude(x_value, y_value, nsteps)  
	  
	blocker = mh.rotate(blocker, make_rotation_matrix(+np.pi/2, (1,0,0)))
	blocker = mh.shift(blocker, (0, 0, height))
	
	return blocker
	
def PlotBlocker(radius, height):
	nsteps = 30
	x_value = np.linspace(radius, radius+200, nsteps)
	y_value = [1]*nsteps
	
	testt = make.rotate_extrude(x_value, y_value, nsteps)    
	testt = mh.rotate(testt, make_rotation_matrix(+np.pi/2, (1,0,0)))
	testt = mh.shift(testt, (0, 0, height))
	
	test_triangles = testt.get_triangle_centers() 
	test_vertices = testt.assemble() 
	
	X2 = test_vertices[:,:,0].flatten()
	Y2 = test_vertices[:,:,1].flatten()
	Z2 = test_vertices[:,:,2].flatten()
	
	trianglestest = [[3*ii,3*ii+1,3*ii+2] for ii in range(len(X2)/3)]
	triang = Triangulation(X2, Y2, trianglestest)
	
	return triang, Z2
	#ax.plot_trisurf(triang, Z2, color="white", shade = True, alpha = .80)
		
	#plt.show() 

def event_display(numPhotons, photon_track, vertex,surf2, lens2, pos, rep, config):
	fig = plt.figure(figsize=(20, 10))
	ax = fig.gca(projection='3d')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax._axis3don = False
	
	# Get triangle vertices for entire mesh
	lens_vertices = lens2.assemble()
	
	surf_vertices = surf2.assemble()
	
	# Get total number of triangles for the curved surface
	nr_triangles2 = len(surf2.get_triangle_centers()[:,1])
	
	# Get a complete list of the 3 coordiantes seperately 
	X2 = lens_vertices[:,:,0].flatten()
	Y2 = lens_vertices[:,:,1].flatten()
	Z2 = lens_vertices[:,:,2].flatten()
	
	# Construct blocker around the lens. The maximal radial distance is the first parameter and the second is the height at this position. 
	triang_blocker, Z_blocker = PlotBlocker(np.max(X2), Z2[np.argmax(X2)])
	
	# Plot blocker. 
	ax.plot_trisurf(triang_blocker, Z_blocker, color="white", shade = True, alpha = 0.8)

	# Define triangulation, e.g. define which coordinates belong to one triangle, so that the mesh can be build properly. Typically, each pair of three subsequent coordinates in 3D belong to one triangle. 
	triangles = [[3*ii,3*ii+1,3*ii+2] for ii in range(len(X2)/3)]
	
	# Compute triangulation 
	triang = Triangulation(X2, Y2, triangles)
	
	# Plot only half of all the triangles, since we are grouping two triangles to one PMT pixel. 
	for ii in range(nr_triangles2/2):
		ax.plot(surf_vertices[ii,:,0], surf_vertices[ii,:,1], surf_vertices[ii,:,2], color='black', lw = 1.)
	
	# Plot photon rays. 
	for i in range(numPhotons):
		# Convert the seperate coordinates into lists.
		X,Y,Z = photon_track[:,i,0].tolist(), photon_track[:,i,1].tolist(), photon_track[:,i,2].tolist()
		
		# Insert the starting point of each photon. 
		X.insert(0, pos[i,0])
		Y.insert(0, pos[i,1])
		Z.insert(0, pos[i,2])
		
		# Plot rays as red lines. 
		ax.plot(X,Y,Z, linewidth=2, color='r')
		
		# Additionally, one could also plot the points of refraction as small black dots. 	
		ax.scatter(X,Y,Z , c='k', s=0.)		
		
	# Specify view, that suits the purpose of visulizing how the design works.
	ax.view_init(elev=15, azim=90)
	
	#Plot the lens with the previously defined triangulation and make it a bit transparent with alpha < 1. 
	ax.plot_trisurf(triang, Z2, color="white", shade = True, alpha = .80)
	
	plt.show()
		
def photon_angle(pos, rep):
    off = (rep-1)*100/2
    photon_pos = np.zeros((rep*rep,3))
    upshift = 800
    for i in range(rep):
		photon_pos[i,:] = [pos[0]-i*100+off-100, pos[1]+off+400, pos[2]+upshift]      
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
    
		
    
