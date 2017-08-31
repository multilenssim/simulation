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

import kabamland2 as kb 


def build_kabamland(kabamland, configname):
    # focal_length sets dist between lens plane and PMT plane (or back of curved detecting surface);
    #(need not equal true lens focal length)
    config = detectorconfig.configdict[configname]
	
    _, lens = kb.get_lens_triangle_centers(config.edge_length, config.base, config.diameter_ratio, config.thickness_ratio, config.half_EPD, config.blockers, blocker_thickness_ratio=config.blocker_thickness_ratio, light_confinement=config.light_confinement, focal_length=config.focal_length, lens_system_name=config.lens_system_name)
    surf = mh.rotate(kb.curved_surface2(config.detector_r, diameter=2*kb.find_max_radius(config.edge_length, config.base), nsteps=9), make_rotation_matrix(-np.pi/2, (1,0,0)))
    surf = mh.shift(surf, (0, 0, -config.focal_length))
    kabamland.add_solid(Solid(surf, lm.ls, lm.ls, lm.fulldetect, 0x0000FF), rotation=None, displacement=(0,0,0))
    kabamland.add_solid(Solid(lens, lm.lensmat, lm.ls) , rotation=None, displacement=None)
    
    blocker = BuildBlocker(np.max(lens.assemble()[:,:,0].flatten()), lens.assemble()[:,:,2].flatten()[np.argmax(lens.assemble()[:,:,0].flatten())])
    kabamland.add_solid(Solid(blocker, lm.ls, lm.ls, lm.fullabsorb, 0x0000FF), rotation=None, displacement=(0,0,0))
    kabamland.channel_index_to_channel_id = [1]
    return lens, surf
    
def getRandom():
	nthreads_per_block = 64
	max_blocks = 1024
	rng_states = gpu.get_rng_states(nthreads_per_block*max_blocks, seed=0)
	doRandom = [nthreads_per_block, max_blocks, rng_states]
	return doRandom  
	
def BuildBlocker(radius, height): 
	nsteps = 50
	x_value = np.linspace(radius, radius+200, nsteps)
	y_value = [1]*nsteps
	
	blocker = make.rotate_extrude(x_value, y_value, nsteps)  
	  
	blocker = mh.rotate(blocker, make_rotation_matrix(+np.pi/2, (1,0,0)))
	blocker = mh.shift(blocker, (0, 0, height))
	
	return blocker
	
def PlotBlocker(radius, height):
	nsteps = 30
	x_value = np.linspace(radius, radius+200, nsteps)
	y_value = [0]*nsteps
	
	blocker = make.rotate_extrude(x_value, y_value, nsteps)    
	blocker = mh.rotate(blocker, make_rotation_matrix(+np.pi/2, (1,0,0)))
	blocker = mh.shift(blocker, (0, 0, height))
	
	blocker_triangles = blocker.get_triangle_centers() 
	blocker_vertices = blocker.assemble() 
	
	X = blocker_vertices[:,:,0].flatten()
	Y = blocker_vertices[:,:,1].flatten()
	Z = blocker_vertices[:,:,2].flatten()
	
	trianglesblocker = [[3*ii,3*ii+1,3*ii+2] for ii in range(len(X)/3)]
	triang = Triangulation(X, Y, trianglesblocker)
	
	return triang, Z

def event_display(numPhotons, photon_track, vertex, surf2, lens2, pos):
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
	
	# Get a complete list of the 3 coordinates seperately 
	X2 = lens_vertices[:,:,0].flatten()
	Y2 = lens_vertices[:,:,1].flatten()
	Z2 = lens_vertices[:,:,2].flatten()
	
	# Construct blocker around the lens. The maximal radial distance is the first parameter and the second is the height at this position. 
	triang_blocker, Z_blocker = PlotBlocker(np.max(X2), Z2[np.argmax(X2)])
	
	# Plot blocker. 
	ax.plot_trisurf(triang_blocker, Z_blocker, color="grey", alpha = 0.3)

	# Define triangulation, e.g. define which coordinates belong to one triangle, so that the mesh can be build properly. Typically, each pair of three subsequent coordinates in 3D belong to one triangle. 
	triangles = [[3*ii,3*ii+1,3*ii+2] for ii in range(len(X2)/3)]
	
	# Compute triangulation 
	triang = Triangulation(X2, Y2, triangles)
	
	# Plot only half of all the triangles, since we are grouping two triangles to one PMT pixel. 
	for sv in surf_vertices:
		ax.plot(sv[:,0], sv[:,1], sv[:,2], color='black', lw = 1.)
		ax.plot(sv[0:2,0], sv[0:2,1], sv[0:2,2], color='white', lw = 1.2)
	
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
	ax.view_init(elev=12, azim=90)
	
	#Plot the lens with the previously defined triangulation and make it a bit transparent with alpha < 1. 
	ax.plot_trisurf(triang, Z2, color="white", shade = True, alpha = 0.05)
	
	plt.show()
		
def photon_angle(rep, pos=[0,0,0]):
    off = 200
    photon_pos = np.zeros((rep,3))
    upshift = 800
    for i in range(5):
		photon_pos[i,:] = [pos[0]-i*100+off-100, pos[1]+off+400, pos[2]+upshift]
    photon_pos[5,:] = [-800,600,800]
    photon_dir = np.tile(np.array([0.2,-0.55,-1]),(rep,1))
    pol = np.cross(photon_dir, uniform_sphere(rep))
    wavelength = np.repeat(300.0, rep)
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

    config = "cfJiani3_2"
    #config = "cfSam1_1"
    
    kabamland = Detector(lm.ls)
    lens, surf = build_kabamland(kabamland, config)
    kabamland.flatten()
    kabamland.bvh = load_bvh(kabamland)
    #view(kabamland)
    #quit()
    sim = Simulation(kabamland, geant4_processes=0)
    runs = 1
    numPhotons = 6
    events, pos = photon_angle(rep=numPhotons)
    nr_hits = np.zeros((runs))
    doRandom = getRandom()
    for ii in range(runs):
		photons, photon_track = propagate_photon(events, numPhotons, 20, kabamland, doRandom[0], doRandom[1], doRandom[2])
		detected = (photons.flags & (0x1 << 2)).astype(bool) 
		nr_hits[ii] = len(photons.pos[detected])
		print "  Number of photons detected			", nr_hits[ii]
    
    vertex = photons.pos[detected] 
    event_display(numPhotons, photon_track, vertex, surf, lens, pos)
    
		
    
