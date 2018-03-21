from chroma.geometry import Geometry, Material, Mesh, Solid, Surface
from chroma.transform import make_rotation_matrix, normalize
from chroma.detector import Detector, G4DetectorParameters
from chroma.demo.optics import glass, black_surface
from ShortIO.root_short import ShortRootWriter
from chroma.sample import uniform_sphere
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
from chroma import make, view, sample
from contextlib import contextmanager
from chroma.generator import vertex
from chroma.loader import load_bvh
from chroma.sim import Simulation
import detectorconfig, lenssystem
from chroma.pmt import build_pmt
from chroma.event import Photons
import matplotlib.pyplot as plt
import lensmaterials as lm
import meshhelper as mh
import numpy as np
import pickle, os
import paths

inputn = 16.0

def lens(diameter, thickness, nsteps=inputn):
    #constructs a parabolic lens
    a = np.linspace(0, diameter/2, nsteps/2, endpoint=False)
    b = np.linspace(diameter/2, 0, nsteps/2)
    return make.rotate_extrude(np.concatenate((a, b)), np.concatenate((2*thickness/diameter**2*(a)**2-0.5*thickness, -2.0*thickness/diameter**2*(b)**2+0.5*thickness)), nsteps=inputn)

def rot_axis(norm,r_norm):
        if not np.array_equal(norm,r_norm):
                norm = np.broadcast_to(norm,r_norm.shape)
        norm = normalize(norm)
        r_norm = normalize(r_norm)
        axis = normalize(np.cross(norm,r_norm))
        phi = np.arccos(np.einsum('ij,ij->i',norm,r_norm))
        return phi, axis

def cylindrical_shell(inner_radius, outer_radius, thickness, nsteps=inputn):
    #make sure that nsteps is the same as that of rotate extrude in lens
    #inner_radius must be less than outer_radius
    return make.rotate_extrude([inner_radius, outer_radius, outer_radius, inner_radius], [-thickness/2.0, -thickness/2.0, thickness/2.0, thickness/2.0], nsteps)

                         
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
    inscribed_radius = edge_length
    radius_root = inscribed_radius*np.power(np.random.rand(n),1.0/3.0)
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

def plot_mesh_object(mesh, centers=[[0,0,0]]): 
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca(projection='3d')
	
    centers = np.array(centers) 
	
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
	
    vertices = mesh.assemble() 
    X = vertices[:,:,0].flatten()
    Y = vertices[:,:,1].flatten()
    Z = vertices[:,:,2].flatten()
	
    triangles = [[3*ii,3*ii+1,3*ii+2] for ii in range(len(X)/3)]
    triang = Triangulation(X, Y, triangles)
	
    ax.plot_trisurf(triang, Z, color="white", edgecolor="black", shade = True, alpha = 1.0)
	
    plt.show()
	
def get_assembly_xyz(mesh): 
    vertices = mesh.assemble()
    X = vertices[:,:,0].flatten()
    Y = vertices[:,:,1].flatten()
    Z = vertices[:,:,2].flatten()
    return X, Y, Z 
	
def get_lens_triangle_centers(vtx, rad, diameter_ratio, thickness_ratio, half_EPD, blockers=True, blocker_thickness_ratio=1.0/1000, light_confinement=False, focal_length=1.0, lens_system_name=None):
	"""input edge length of icosahedron 'edge_length', the number of small triangles in the base of each face 'base', the ratio of the diameter of each lens to the maximum diameter possible 'diameter_ratio' (or the fraction of the default such ratio, if a curved detector lens system), the ratio of the thickness of the lens to the chosen (not maximum) diameter 'thickness_ratio', the radius of the blocking entrance pupil 'half_EPD', and the ratio of the thickness of the blockers to that of the lenses 'blocker_thickness_ratio' to return the icosahedron of lenses in kabamland. Light_confinment=True adds cylindrical shells behind each lens that absorb all the light that touches them, so that light doesn't overlap between lenses. If lens_system_name is a string that matches one of the lens systems in lenssystem.py, the corresponding lenses and detectors will be built. Otherwise, a default simple lens will be built, with parameters hard-coded below."""
	# Get the list of lens meshes from the appropriate lens system as well as the lens material
	scale_rad = rad*diameter_ratio
	lenses = lenssystem.get_lens_mesh_list(lens_system_name, scale_rad)
	lensmat = lenssystem.get_lens_material(lens_system_name)
	lens_mesh = None
	lens_centers = [] 	
	for lns in lenses: # Add all the lenses for the first lens system to solid 'face'
		if not lens_mesh:
			lens_mesh = lns
		else:
			lens_mesh += lns
	X, Y, Z = get_assembly_xyz(lens_mesh)
	lens_centers = np.asarray([np.mean(X), np.mean(Y), np.mean(Z)])
	lns_center_arr = []
        phi, axs = rot_axis([0,0,1],vtx)
        for vx,ph,ax in zip(vtx,-phi,axs):
		lns_center_arr.append(np.dot(make_rotation_matrix(ph,ax),lens_centers) - vx)
	return np.asarray(lns_center_arr)
	

  
def build_lens_icosahedron(kabamland, vtx, rad, diameter_ratio, thickness_ratio, half_EPD, blockers=True, blocker_thickness_ratio=1.0/1000, light_confinement=False, focal_length=1.0, lens_system_name=None):
    """input edge length of icosahedron 'edge_length', the number of small triangles in the base of each face 'base', the ratio of the diameter of each lens to the maximum diameter possible 'diameter_ratio' (or the fraction of the default such ratio, if a curved detector lens system), the ratio of the thickness of the lens to the chosen (not maximum) diameter 'thickness_ratio', the radius of the blocking entrance pupil 'half_EPD', and the ratio of the thickness of the blockers to that of the lenses 'blocker_thickness_ratio' to return the icosahedron of lenses in kabamland. Light_confinment=True adds cylindrical shells behind each lens that absorb all the light that touches them, so that light doesn't overlap between lenses. If lens_system_name is a string that matches one of the lens systems in lenssystem.py, the corresponding lenses and detectors will be built. Otherwise, a default simple lens will be built, with parameters hard-coded below.
    """
	# Get the list of lens meshes from the appropriate lens system as well as the lens material'''
    scale_rad = rad*diameter_ratio #max_radius->rad of the lens assembly
    lenses = lenssystem.get_lens_mesh_list(lens_system_name, scale_rad)
    lensmat = lenssystem.get_lens_material(lens_system_name)
    face = None
    for lns in lenses:
    	#lns = mh.rotate(lns,make_rotation_matrix(ph,ax)) 
        if not face:
        	face = Solid(lns, lensmat, kabamland.detector_material)
       	else:
        	face += Solid(lns, lensmat, kabamland.detector_material)

    if light_confinement:
	shield = mh.rotate(cylindrical_shell(rad*(1 - 0.001), rad, focal_length,32), make_rotation_matrix(np.pi/2.0, (1,0,0)))
	baffle = Solid(shield, lensmat, kabamland.detector_material, black_surface, 0xff0000)

    if blockers:
    	blocker_thickness = 2*rad*blocker_thickness_ratio
    	if half_EPD < rad:
		c1 = lenssystem.get_lens_sys(lens_system_name).c1*lenssystem.get_scale_factor(lens_system_name,scale_rad)
		offset = [0,0,c1-np.sqrt(c1*c1-rad*rad)]
        	anulus_blocker = mh.shift(mh.rotate(cylindrical_shell(half_EPD, rad, blocker_thickness, 32), make_rotation_matrix(np.pi/2.0, (1,0,0))),offset)
		face += Solid(anulus_blocker, lensmat, kabamland.detector_material, black_surface, 0xff0000)
    phi, axs = rot_axis([0,0,1],vtx)
    for vx,ph,ax in zip(vtx,-phi,axs):
        kabamland.add_solid(face, rotation=make_rotation_matrix(ph,ax), displacement = -vx)
	if light_confinement:
	        kabamland.add_solid(baffle, rotation=make_rotation_matrix(ph,ax), displacement = -normalize(vx)*(np.linalg.norm(vx)+focal_length/2.0))


def calc_steps(x_value,y_value,detector_r,base_pixel):
	x_coord = np.asarray([x_value,np.roll(x_value,-1)]).T[:-1]
	y_coord = np.asarray([y_value,np.roll(y_value,-1)]).T[:-1]
	lat_area = 2*np.pi*detector_r*(y_coord[:,0]-y_coord[:,1])
	n_step = (lat_area/lat_area[-1]*base_pixel).astype(int)
	return x_coord, y_coord, n_step
    
def curved_surface2(detector_r=2.0, diameter = 2.5, nsteps=8,base_pxl=4,ret_arr=False):
    '''Builds a curved surface based on the specified radius. Origin is center of surface.'''
    if (detector_r < diameter/2.0):
        raise Exception('The Radius of the curved surface must be larger than diameter/2.0')
    shift1 = np.sqrt(detector_r**2 - (diameter/2.0)**2)
    theta1 = np.arctan(shift1/(diameter/2.0))
    angles1 = np.linspace(theta1, np.pi/2, nsteps)
    x_value = abs(detector_r*np.cos(angles1))
    y_value = detector_r-detector_r*np.sin(angles1)
    surf = None 
    x_coord,y_coord,n_step = calc_steps(x_value,y_value,detector_r,base_pixel=base_pxl)
    for i,(x,y,n_stp) in enumerate(zip(x_coord,y_coord,n_step)):
	if i == 0:
		surf = make.rotate_extrude(x,y,n_stp)
	else:
		surf += make.rotate_extrude(x,y,n_stp)
    if ret_arr: return  surf, n_step
    else: return surf

def get_curved_surf_triangle_centers(vtx, rad, detector_r = 1.0, focal_length=1.0, nsteps = 10, b_pxl=4):
    #Changed the rotation matrix to try and keep the curved surface towards the interior
    #Make sure diameter, etc. are set properly
    curved_surf_triangle_centers = []
    mesh_surf, ring = curved_surface2(detector_r, diameter=2*rad, nsteps=nsteps, base_pxl=b_pxl,ret_arr=True)
    initial_curved_surf = mh.rotate(mesh_surf, make_rotation_matrix(-np.pi/2, (1,0,0)))     #-np.pi with curved_surface2
    triangles_per_surface = initial_curved_surf.triangles.shape[0]
    phi, axs = rot_axis([0,0,1],vtx)
    for vx,ph,ax in zip(vtx,-phi,axs):
	curved_surf_triangle_centers.extend(mh.shift(mh.rotate(initial_curved_surf,make_rotation_matrix(ph,ax)),-normalize(vx)*(np.linalg.norm(vx)+focal_length)).get_triangle_centers())
    return np.asarray(curved_surf_triangle_centers),triangles_per_surface,ring

def build_curvedsurface_icosahedron(kabamland, vtx, rad, diameter_ratio, focal_length=1.0, detector_r = 1.0, nsteps = 10, b_pxl=4):
    initial_curved_surf = mh.rotate(curved_surface2(detector_r, diameter=rad*2, nsteps=nsteps, base_pxl=b_pxl), make_rotation_matrix(-np.pi/2, (1,0,0)))
    face = Solid(initial_curved_surf, kabamland.detector_material, kabamland.detector_material, lm.fulldetect, 0x0000FF)
    phi, axs = rot_axis([0,0,1],vtx)
    for vx,ph,ax in zip(vtx,-phi,axs):
        kabamland.add_solid(face, rotation=make_rotation_matrix(ph,ax), displacement = -normalize(vx)*(np.linalg.norm(vx)+focal_length))


def build_pmt_icosahedron(kabamland, vtx, focal_length=1.0):
    offset = 1.2*(vtx+focal_length)
    angles = np.linspace(np.pi/4, 2*np.pi+np.pi/4, 4, endpoint=False)
    square = make.linear_extrude(offset*np.sqrt(2)*np.cos(angles),offset*np.sqrt(2)*np.sin(angles),2.0)
    vrs = np.eye(3)
    for vr in vrs:
	if np.array_equal(vr,[0,0,1]):
		kabamland.add_pmt(Solid(square, glass, kabamland.detector_material, lm.fullabsorb, 0xBBFFFFFF), displacement = offset*vr)
		kabamland.add_pmt(Solid(square, glass, kabamland.detector_material, lm.fullabsorb, 0xBBFFFFFF), displacement = -offset*vr)
	else:
		trasl = np.cross(vr,[0,0,1])
    		kabamland.add_pmt(Solid(square, glass, kabamland.detector_material, lm.fullabsorb, 0xBBFFFFFF), rotation = make_rotation_matrix(np.pi/2,vr), displacement = offset*trasl)
		kabamland.add_pmt(Solid(square, glass, kabamland.detector_material, lm.fullabsorb, 0xBBFFFFFF), rotation = make_rotation_matrix(np.pi/2,vr), displacement = -offset*trasl)

def build_kabamland(kabamland, configname):
    # focal_length sets dist between lens plane and PMT plane (or back of curved detecting surface);
    #(need not equal true lens focal length)
    config = detectorconfig.configdict(configname)
    build_lens_icosahedron(kabamland, config.vtx, config.half_EPD/config.EPD_ratio, config.diameter_ratio, config.thickness_ratio, config.half_EPD, config.blockers, blocker_thickness_ratio=config.blocker_thickness_ratio, light_confinement=config.light_confinement, focal_length=config.focal_length, lens_system_name=config.lens_system_name)
    build_curvedsurface_icosahedron(kabamland, config.vtx, config.half_EPD/config.EPD_ratio, config.diameter_ratio, focal_length=config.focal_length, detector_r=config.detector_r, nsteps=config.nsteps, b_pxl=config.b_pixel)
    build_pmt_icosahedron(kabamland, np.linalg.norm(config.vtx[0]), focal_length=config.focal_length) # Built further out, just as a way of stopping photons    

def driver_funct(configname):
	kabamland = Detector(lm.create_scintillation_material())
	config = detectorconfig.configdict(configname)
	#get_lens_triangle_centers(vtx, rad_assembly, config.diameter_ratio, config.thickness_ratio, config.half_EPD, config.blockers, blocker_thickness_ratio=config.blocker_thickness_ratio, light_confinement=config.light_confinement, focal_length=config.focal_length, lens_system_name=config.lens_system_name)
	#print get_curved_surf_triangle_centers(config.vtx, config.half_EPD/config.EPD_ratio, config.detector_r, config.focal_length, config.nsteps, config.b_pixel)[0]
	build_lens_icosahedron(kabamland, config.vtx, config.half_EPD/config.EPD_ratio, config.diameter_ratio, config.thickness_ratio, config.half_EPD, config.blockers, blocker_thickness_ratio=config.blocker_thickness_ratio, light_confinement=config.light_confinement, focal_length=config.focal_length, lens_system_name=config.lens_system_name)
	#build_curvedsurface_icosahedron(kabamland, config.vtx, config.half_EPD/config.EPD_ratio, config.diameter_ratio, focal_length=config.focal_length, detector_r=config.detector_r, nsteps=config.nsteps, b_pxl=config.b_pixel)
	#build_pmt_icosahedron(kabamland, np.linalg.norm(config.vtx[0]), focal_length=config.focal_length)
	kabamland.flatten()
	kabamland.bvh = load_bvh(kabamland)
	view(kabamland)

def full_detector_simulation(amount, configname, simname, datadir="",cal=''):
        #simulates 1000*amount photons uniformly spread throughout a sphere whose radius is the inscribed radius of the icosahedron. Note that viewing may crash if there are too many lenses. (try using configview)

        config = detectorconfig.configdict(configname)
        print('Starting to build')
        g4_detector_parameters=G4DetectorParameters(orb_radius=7., world_material='G4_Galactic')
        kabamland = load_or_build_detector(configname, lm.create_scintillation_material(), g4_detector_parameters=g4_detector_parameters)
        print('Detector was built')

        f = ShortRootWriter(datadir + simname)
        sim = Simulation(kabamland,geant4_processes=0)
	scale_factor = 1
	if cal == '_narrow':
		scale_factor = 0.2
        for j in range(100):
                print j
                sim_events = [uniform_photons(config.edge_length*scale_factor, amount) for i in range(10)]
                for ev in sim.simulate(sim_events, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
                        f.write_event(ev)
        f.close()

def load_or_build_detector(config, detector_material, g4_detector_parameters):
    filename = paths.detector_pickled_path + config + '.pickle'
    if not os.path.exists(paths.detector_pickled_path):
        os.makedirs(paths.detector_pickled_path)
    # How to ensure the material and detector parameters are correct??
    try:
        with open(filename,'rb') as pickle_file:
            print("** Loading detector configuration: " + config)
            kabamland = pickle.load(pickle_file)
            pickle_has_g4_dp = hasattr(kabamland, 'g4_detector_parameters') and kabamland.g4_detector_parameters is not None
            pickle_has_g4_dm = hasattr(kabamland, 'detector_material') and kabamland.detector_material is not None
            if g4_detector_parameters is not None:
                print('*** Using Geant4 detector parameters specified' +
                      (' - replacement' if pickle_has_g4_dp else '') + ' ***')
                kabamland.g4_detector_parameters = g4_detector_parameters
            elif pickle_has_g4_dp:
                print('*** Using Geant4 detector parameters found in loaded file ***')
            else:
                print('*** No Geant4 detector parameters found at all ***')

            if detector_material is not None:
                print('*** Using Geant4 detector material specified' +
                      (' - replacement' if pickle_has_g4_dm else '') + ' ***')
                kabamland.detector_material = detector_material
            elif pickle_has_g4_dm:
                print('*** Using Geant4 detector material found in loaded file ***')
            else:
                print('*** No Geant4 detector material found at all ***')
    except IOError as error:
        print("** Building detector configuration: " + config)
        kabamland = Detector(lm.create_scintillation_material(), g4_detector_parameters=g4_detector_parameters)
        build_kabamland(kabamland, config)
        kabamland.flatten()
        kabamland.bvh = load_bvh(kabamland)
        try:
            with open(filename,'wb') as pickle_file:
                pickle.dump(kabamland, pickle_file)
        except IOError as error:
            print("Error writing pickle file: " + filename)
    return kabamland


if __name__ == '__main__':
	driver_funct('cfSam1_K20_8_small')
