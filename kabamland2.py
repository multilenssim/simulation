import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse

from chroma.geometry import Solid
from chroma.transform import make_rotation_matrix, normalize
from chroma.demo.optics import glass, black_surface
from chroma.detector import Detector
from chroma.sample import uniform_sphere
from chroma import make, sample
from chroma.event import Photons
from chroma.camera import view

import detectorconfig, lenssystem
import lensmaterials as lm
import meshhelper as mh

inputn = 16.0

def lens(diameter, thickness, nsteps=inputn):
    #constructs a parabolic lens
    a = np.linspace(0, diameter/2, nsteps/2, endpoint=False)
    b = np.linspace(diameter/2, 0, nsteps/2)
    return make.rotate_extrude(np.concatenate((a, b)), np.concatenate((2*thickness/diameter**2*(a)**2-0.5*thickness,
                                                                       -2.0*thickness/diameter**2*(b)**2+0.5*thickness)), nsteps=inputn)

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

def uniform_photons(inscribed_radius, n):
    #constructs photons uniformly throughout the detector inside of the inscribed sphere.
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

def constant_photons(pos, n):
    #constructs photons at one location with random propagation directions
    points = np.empty((n,3))
    points[:] = pos
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
	
def get_lens_triangle_centers(vtx, rad, diameter_ratio, lens_system_name=None):
    # TODO: The comment here was a copy of the comment for build_lens_icosahedron.  Needs a new description
    scale_rad = rad*diameter_ratio
    lens_config = lenssystem.get_lens_sys(lens_system_name)
    lenses = lens_config.get_lens_mesh_list(scale_rad)  # TODO: this should come from the detector file so that we don't have to recalculate
    lens_mesh = None
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

# TODO: get rid of the default focal length?  Or the focal length at all?  It's only used to place the baffles
def build_lens_icosahedron(kabamland, vtx, rad, diameter_ratio, thickness_ratio, half_EPD, blockers=True,
                           blocker_thickness_ratio=1.0/1000, light_confinement=False, pmt_surface_position=1.0, lens_system_name=None):
    """input edge length of icosahedron 'edge_length', the number of small triangles in the base of each face 'base',
       the ratio of the diameter of each lens to the maximum diameter possible 'diameter_ratio' (or the fraction of the default such ratio,
       if a curved detector lens system), the ratio of the thickness of the lens to the chosen (not maximum) diameter 'thickness_ratio',
       the radius of the blocking entrance pupil 'half_EPD', and the ratio of the thickness of the blockers to that of the lenses 'blocker_thickness_ratio'
       to return the icosahedron of lenses in kabamland. Light_confinment=True adds cylindrical shells behind each lens that
       absorb all the light that touches them, so that light doesn't overlap between lenses.
       If lens_system_name is a string that matches one of the lens systems in lenssystem.py, the corresponding lenses and detectors will be built.
       Otherwise, a default simple lens will be built, with parameters hard-coded below.
    """
    # Get the list of lens meshes from the appropriate lens system as well as the lens material
    lens_config = lenssystem.get_lens_sys(lens_system_name)
    scale_rad = rad*diameter_ratio #max_radius->rad of the lens assembly
    lenses = lens_config.get_lens_mesh_list(scale_rad)
    lensmat = lens_config.get_lens_material()
    face = None
    element_vertex_offsets = [0]  # Track the number of vertices in each component for plotting purposes
    for lns in lenses:
        #lns = mh.rotate(lns,make_rotation_matrix(ph,ax))
        if not face:
            face = Solid(lns, lensmat, kabamland.detector_material)
        else:
            face += Solid(lns, lensmat, kabamland.detector_material)
        element_vertex_offsets.append(len(face.mesh.vertices))

    # TODO cleanup:
    #   When we need which variables and when we have which variables in the config
    #   get rid of focal length - it's not the right dimension for the baffle length anyway - it needs to be the EPD to focal array distance.
    #   See below
    scale_factor = lens_config.get_scale_factor(scale_rad)
    epd_offset = [0,0,lens_config.epd_offset*scale_factor]
    baffle_offset = [0, 0, lens_config.epd_offset * scale_factor - pmt_surface_position / 2]  # lens_config.epd_offset*scale_factor-1000]
    if light_confinement:
        shield = mh.shift(mh.rotate(cylindrical_shell(rad * (1 - 0.001), rad, pmt_surface_position, 32), make_rotation_matrix(np.pi / 2.0, (1, 0, 0))), baffle_offset)
        baffle = Solid(shield, lensmat, kabamland.detector_material, black_surface, 0xff0000)

    if blockers:
        blocker_thickness = 2*rad*blocker_thickness_ratio
        if half_EPD < rad:
            c1 = lens_config._c1 * scale_factor   # TODO XXXXX: make an accessor for c1
            z_ofst = c1-np.sqrt(c1*c1-rad*rad)
            correct_offset = z_ofst + 35.3
            print('curvature: %f, radius: %f (of what), scale_factor: %f, initial offset: %f, correct offset: %f, new offset: %f' %
                  (c1, rad, scale_factor, z_ofst, correct_offset, lens_config.epd_offset*scale_factor))
            print('Half EPD: %f' % half_EPD)
            anulus_blocker = mh.shift(mh.rotate(cylindrical_shell(half_EPD, rad, blocker_thickness, 32), make_rotation_matrix(np.pi/2.0, (1,0,0))),epd_offset)
            face += Solid(anulus_blocker, lensmat, kabamland.detector_material, black_surface, 0xff0000)
            element_vertex_offsets.append(len(face.mesh.vertices))
            #face += baffle  # TODO: Just add to plot baffle - so the baffle is added twice? - right now in different places...
            #element_vertex_offsets.append(len(face.mesh.vertices))

    phi, axs = rot_axis([0,0,1],vtx)
    for vx,ph,ax in zip(vtx,-phi,axs):
        kabamland.add_solid(face, rotation=make_rotation_matrix(ph,ax), displacement = -vx)
        if light_confinement:
            kabamland.add_solid(baffle, rotation=make_rotation_matrix(ph,ax), displacement = -normalize(vx)*(np.linalg.norm(vx) - pmt_surface_position / 2.0))
    return element_vertex_offsets


def calc_steps(x_value, y_value, detector_r, base_pixel):
    x_coord = np.asarray([x_value, np.roll(x_value, -1)]).T[:-1]
    y_coord = np.asarray([y_value, np.roll(y_value, -1)]).T[:-1]
    lat_area = 2 * np.pi * detector_r * (y_coord[:, 0] - y_coord[:, 1])
    n_step = (lat_area / lat_area[-1] * base_pixel).astype(int)
    # print('Pixel areas per ring: %s' % str(lat_area / n_step))
    # print('Coords: %s %s' % (x_coord, y_coord))
    # print('Values: %s %s' % (x_value, y_value))
    return x_coord, y_coord, n_step


def curved_surface2(detector_r=2.0, diameter=2.5, nsteps=8, base_pxl=4, ret_arr=False):
    '''Builds a curved surface based on the specified radius. Origin is center of surface.'''
    if (detector_r < diameter / 2.0):
        raise Exception('The Radius of the curved surface must be larger than diameter/2.0')
    shift1 = np.sqrt(detector_r ** 2 - (diameter / 2.0) ** 2)
    theta1 = np.arctan(shift1 / (diameter / 2.0))
    angles1 = np.linspace(theta1, np.pi / 2, nsteps)
    # print('Parameters: %f %f %s' % (shift1, theta1, str(angles1)))
    x_value = abs(detector_r * np.cos(angles1))
    y_value = detector_r - detector_r * np.sin(angles1)
    surf = None
    x_coord, y_coord, n_step = calc_steps(x_value, y_value, detector_r, base_pixel=base_pxl)
    for i, (x, y, n_stp) in enumerate(zip(x_coord, y_coord, n_step)):
        if i == 0:
            surf = make.rotate_extrude(x, y, n_stp)
        else:
            surf += make.rotate_extrude(x, y, n_stp)
    if ret_arr:
        return surf, n_step
    else:
        return surf


def get_curved_surf_triangle_centers(vtx, rad, detector_r, pmt_surface_position, nsteps=10, b_pxl=4):
    # Changed the rotation matrix to try and keep the curved surface towards the interior
    # Make sure diameter, etc. are set properly
    curved_surf_triangle_centers = []
    mesh_surf, ring = curved_surface2(detector_r, diameter=2 * rad, nsteps=nsteps, base_pxl=b_pxl, ret_arr=True)
    initial_curved_surf = mh.rotate(mesh_surf, make_rotation_matrix(-np.pi / 2, (1, 0, 0)))  # -np.pi with curved_surface2
    triangles_per_surface = initial_curved_surf.triangles.shape[0]
    phi, axs = rot_axis([0, 0, 1], vtx)
    for vx, ph, ax in zip(vtx, -phi, axs):
        curved_surf_triangle_centers.extend(mh.shift(mh.rotate(initial_curved_surf, make_rotation_matrix(ph, ax)),
                                                     -normalize(vx) * (np.linalg.norm(vx) + pmt_surface_position)).get_triangle_centers())
    return np.asarray(curved_surf_triangle_centers), triangles_per_surface, ring

def build_curvedsurface_icosahedron(kabamland, vtx, rad, pmt_surface_position, detector_r, nsteps = 10, b_pxl=4):
    initial_curved_surf = mh.rotate(curved_surface2(detector_r, diameter=rad*2, nsteps=nsteps, base_pxl=b_pxl), make_rotation_matrix(-np.pi/2, (1,0,0)))
    face = Solid(initial_curved_surf, kabamland.detector_material, kabamland.detector_material, lm.fulldetect, 0x0000FF)
    phi, axs = rot_axis([0,0,1],vtx)
    for vx,ph,ax in zip(vtx,-phi,axs):
        kabamland.add_solid(face, rotation=make_rotation_matrix(ph,ax), displacement = -normalize(vx)*(np.linalg.norm(vx)+pmt_surface_position))


def build_pmt_icosahedron(kabamland, vtx, pmt_surface_position):
    offset = 1.2*(vtx+pmt_surface_position)
    angles = np.linspace(np.pi/4, 2*np.pi+np.pi/4, 4, endpoint=False)
    square = make.linear_extrude(offset*np.sqrt(2)*np.cos(angles),offset*np.sqrt(2)*np.sin(angles),2.0)
    vrs = np.eye(3)
    for vr in vrs:
        if np.array_equal(vr,[0,0,1]):
            kabamland.add_pmt(Solid(square, glass, kabamland.detector_material, lm.fullabsorb, 0xBBFFFFFF), displacement = offset*vr)
            kabamland.add_pmt(Solid(square, glass, kabamland.detector_material, lm.fullabsorb, 0xBBFFFFFF), displacement = -offset*vr)
        else:
            trasl = np.cross(vr,[0,0,1])
            kabamland.add_pmt(Solid(square, glass, kabamland.detector_material, lm.fullabsorb, 0xBBFFFFFF),
                                  rotation = make_rotation_matrix(np.pi/2,vr), displacement = offset*trasl)
            kabamland.add_pmt(Solid(square, glass, kabamland.detector_material, lm.fullabsorb, 0xBBFFFFFF),
                              rotation = make_rotation_matrix(np.pi/2,vr), displacement = -offset*trasl)

def build_kabamland(kabamland, config):
    # pmt_surface_position / focal_length sets dist between lens plane and PMT plane (or back of curved detecting surface);
    # (need not equal true lens focal length)

    # TODO: These are not really building the icosahedron right?
    # TODO: just pass the config
    #build_lens_icosahedron(kabamland, config.vtx, config.half_EPD/config.EPD_ratio,
    #       config.diameter_ratio, config.thickness_ratio, config.half_EPD, config.blockers,
    #       blocker_thickness_ratio=config.blocker_thickness_ratio, light_confinement=config.light_confinement,
    #       pmt_surface_position=config.pmt_surface_position, lens_system_name=config.lens_system_name)
    #build_curvedsurface_icosahedron(kabamland, config.vtx, config.half_EPD/config.EPD_ratio,
    #       config.diameter_ratio, pmt_surface_position=config.pmt_surface_position,
    #       detector_r=config.detector_r, nsteps=config.ring_count, b_pxl=config.base_pixels)
    build_lens_icosahedron(kabamland, config.vtx, config.max_radius, config.diameter_ratio,
                           config.thickness_ratio, config.half_EPD, config.blockers,
                           blocker_thickness_ratio=config.blocker_thickness_ratio,
                           light_confinement=config.light_confinement, pmt_surface_position=config.pmt_surface_position,
                           lens_system_name=config.lens_system_name)
    build_curvedsurface_icosahedron(kabamland, config.vtx, config.max_radius,
                                    config.pmt_surface_position, config.detector_r,
                                    nsteps=config.ring_count, b_pxl=config.base_pixels)
    build_pmt_icosahedron(kabamland, np.linalg.norm(config.vtx[0]), config.pmt_surface_position) # Built further out, just as a way of stopping photons

def driver_funct(configname):
    from chroma.loader import load_bvh  # Requires CUDA so only import it when necessary
    kabamland = Detector(lm.create_scintillation_material())
    config = detectorconfig.get_detector_config(configname)
    #get_lens_triangle_centers(vtx, rad_assembly, config.diameter_ratio,
    #       config.thickness_ratio, config.half_EPD, config.blockers,
    #       blocker_thickness_ratio=config.blocker_thickness_ratio,
    #       light_confinement=config.light_confinement, pmt_surface_position=config.pmt_surface_position,
    #       lens_system_name=config.lens_system_name)
    #print get_curved_surf_triangle_centers(config.vtx, config.half_EPD/config.EPD_ratio,
    #       config.detector_r, config.pmt_surface_position, config.nsteps, config.b_pixel)[0]
    element_vertex_offsets = build_lens_icosahedron(kabamland, config.vtx, config.half_EPD / config.EPD_ratio, config.diameter_ratio,
                                                    config.thickness_ratio, config.half_EPD, config.blockers,
                                                    blocker_thickness_ratio=config.blocker_thickness_ratio,
                                                    light_confinement=config.light_confinement, pmt_surface_position=config.pmt_surface_position,
                                                    lens_system_name=config.lens_system_name)
    #build_curvedsurface_icosahedron(kabamland, config.vtx, config.half_EPD/config.EPD_ratio,
    #       config.diameter_ratio, pmt_surface_position=config.pmt_surface_position, detector_r=config.detector_r,
    #       nsteps=config.nsteps, b_pxl=config.b_pixel)
    #build_pmt_icosahedron(kabamland, np.linalg.norm(config.vtx[0]), pmt_surface_position=config.pmt_surface_position)
    kabamland.flatten()
    #kabamland.bvh = load_bvh(kabamland)
    view(kabamland)

    return kabamland.mesh.vertices, element_vertex_offsets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_name', help='detector configuration', nargs='?', default='cfSam2-0.5_l200_p104400_b8_e5')
    _args = parser.parse_args()
    config_name = _args.config_name

    vtx, element_vertex_offsets = driver_funct(config_name)   # 'cfSam2-0.5_l200_p104400_b8_e5')  # ''cfSam1_l200_p107600_b4_e10')  'cfSam1_l200_p96000_b8_e8') #
    print('Vertex shape for %s: %s' % (config_name, str(vtx.shape)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for offset in element_vertex_offsets:
        if offset != 0:
            ax.plot(vtx[last_offset:offset, 0], vtx[last_offset:offset, 1], vtx[last_offset:offset, 2], '.')
        last_offset = offset
    #ax.plot(vtx[:, 0], vtx[:, 1], vtx[:, 2], '.')
    plt.show()
