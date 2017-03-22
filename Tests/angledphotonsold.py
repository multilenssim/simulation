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

#for all lenses, radius must be at least diameter/2.0
def lens(radius, diameter, nsteps=1024):
    #constructs a convex lens mesh
    angles = np.linspace(-np.pi/2, np.pi/2, nsteps)
    return make.rotate_extrude(diameter/2*np.cos(angles), np.sign(angles)*(np.sqrt(radius**2-(diameter/2*np.cos(angles))**2)-np.sqrt(radius**2-(diameter/2)**2)), nsteps=64)

def pclens(radius, diameter, nsteps=1024):
    #constructs a plano-convex lens mesh
    angles = np.linspace(-np.pi/2, np.pi/2, nsteps)
    return make.rotate_extrude(diameter/2.0*np.cos(angles), (0.5*(np.sign(angles)+1))*(np.sqrt(radius**2-(diameter/2*np.cos(angles))**2)-np.sqrt(radius**2-(diameter/2)**2)), nsteps=64)

def paralens(focal_length, diameter, nsteps=1024):
    #constructs a paraboloid lens
    height = 0.25/focal_length*(diameter/2)**2
    a = np.linspace(0, diameter/2, nsteps/2, endpoint=False)
    b = np.linspace(diameter/2, 0, nsteps/2)
    return make.rotate_extrude(np.concatenate((a, b)), np.concatenate((0.25/focal_length*(a)**2-height, -0.25/focal_length*(b)**2+height)), nsteps=64)

def pcparalens(focal_length, diameter, nsteps=1024):
    #constructs a planoconvex paraboloid lens
    height = 0.25/focal_length*(diameter/2)**2
    a = np.linspace(0, diameter/2, nsteps/2, endpoint=False)
    b = np.linspace(diameter/2, 0, nsteps/2)
    c = np.tile(height, nsteps/2)
    return make.rotate_extrude(np.concatenate((a, b)), np.concatenate((0.25/focal_length*(a)**2, c)), nsteps=64)

def cylindrical_shell(inner_radius, outer_radius, height, nsteps=1024):
    if inner_radius < outer_radius:
        return make.rotate_extrude([inner_radius, outer_radius, outer_radius, inner_radius], [-height/2.0, -height/2.0, height/2.0, height/2.0], nsteps)
    else:
        print "inner radius must be less than outer radius"

def photon_rays_angledx(n, diameter, theta):
    #constructs collimated photons traveling at a specified angle from the x-axis in the xy-plane. z_initial = 0, y_initial = 1
    #theta can range from -pi/2 to pi/2
    pos = np.array([(i+np.tan(theta), 1.0, 0) for i in np.linspace(-diameter/2, diameter/2, n)])
    dir = np.tile((-np.sin(theta), -np.cos(theta), 0), (n,1))
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000, n)
    return Photons(pos, dir, pol, wavelengths)

def photon_rays_angledx_circle(n, diameter, theta):
    #constructs collimated photons traveling in a uniform circle at a specified angle from the x-axis in the xy-plane. z_initial = 0, y_initial = 1
    #theta ranges from -pi/2 to pi/2
    radii = np.random.uniform(0, diameter/2, n)
    angles = np.random.uniform(0, 2*np.pi, n)
    points = np.empty((n,3))
    points[:,0] = np.sqrt(diameter/2)*np.sqrt(radii)*np.cos(angles) + np.tan(theta)
    points[:,1] = np.repeat(1.0, n)
    points[:,2] = np.sqrt(diameter/2)*np.sqrt(radii)*np.sin(angles)
    pos = points
    dir = np.tile((-np.tan(theta),-1,0), (n,1))
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000, n)
    return Photons(pos, dir, pol, wavelengths)

def photon_rays_angledz(n, diameter, theta):
    #constructs collimated photons traveling at a specified angle from the y axis in the yz plane. y-initial=1
    #theta ranges from -pi/2 to pi/2
    pos = np.array([(i, 1, np.tan(theta)) for i in np.linspace(-diameter/2, diameter/2, n)])
    dir = np.tile((0,-1,-np.tan(theta)), (n,1))
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000, n)
    return Photons(pos, dir, pol, wavelengths)

def pc_focal_length(radius):
    #using thin lens equation. thick lens focal length equation yields same result (second order term drops out with R2 -> infinity)
    n1 = lm.ls_refractive_index
    n2 = lm.lensmat_refractive_index
    focal_length = 1.0/((n2-n1)*(1.0/radius))
    return focal_length

if __name__ == '__main__':
    from chroma.sim import Simulation
    from chroma import sample
    from chroma.event import Photons
    from chroma.loader import load_bvh
    from chroma.generator import vertex
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # print matplotlib.__version__
    # print matplotlib.__file__
    # print matplotlib.get_configdir()
    # print np.__version__

    pcrad = 0.75
    print pc_focal_length(pcrad)

    #defining the specifications of the meshes
    lens1 = lens(5.0, 1.0)
    lens2 = pclens(pcrad, 1.0)
    lens3 = paralens(0.5, 1.0)
    lens4 = pcparalens(0.5,1.0)
    lens5 = make.box(1.0,1.0,1.0)
    
    #creating solids of form (mesh, inside_mat, outside_mat)
    lens_solid1 = Solid(lens1, lm.lensmat, lm.ls)
    lens_solid2 = Solid(lens2, lm.lensmat, lm.ls)
    #lens_solid2 = Solid(lens2, lm.lensmat, lm.ls, lm.noreflect)
    lens_solid3 = Solid(lens3, lm.lensmat, lm.ls)
    lens_solid4 = Solid(lens4, lm.lensmat, lm.ls)
    lens_solid5 = Solid(lens5, lm.lensmat, lm.ls)
    
    pmt = Solid(make.box(100.0,1.0,100.0), glass, lm.ls, surface=lm.fulldetect, color=0x00ff00)
    blocker = Solid(cylindrical_shell(0.5, 100.0, 0.00001), glass, lm.ls, black_surface, 0xff0000)

    ultraabsorb = Solid(make.box(100, 100, 100), lm.ls, lm.ls, surface = lm.fulldetect)

    #creates the entire Detector and adds the solid & pmt
    ftlneutrino = Detector(lm.ls)
    #ftlneutrino.add_solid(ultraabsorb)
    ftlneutrino.add_solid(lens_solid2)
    ftlneutrino.add_solid(blocker)
    ftlneutrino.add_pmt(pmt, rotation=None, displacement=(0,-0.5-pc_focal_length(pcrad), 0))
    #-2.02276 is the calculated value for paralens(0.5, 1.0)
    #-8.14654 is y value for lens1 with focal length 7.65654, box of sides: 1.0, R=5, d=1, T=0.0501256
    ftlneutrino.flatten()
    ftlneutrino.bvh = load_bvh(ftlneutrino)

    #view(ftlneutrino)
    
    from chroma.io.root import RootWriter
    f = RootWriter('test.root')

    sim = Simulation(ftlneutrino)
   
   
    # ftlphotonsx = photon_rays_angledx(10000.0, 1.0, np.pi/4)
    # ftlphotonsxcir = photon_rays_angledx_circle(10000.0, 1.0, np.pi/4)
    # ftlphotonsz = photon_rays_angledz(10000.0, 1.0, np.pi/4)
    
    num = 50
    angles = np.linspace(0, np.pi/3, num)
    deviations = np.zeros(num)
    #performance_angle represents the angle that first passes a standard deviation of 0.5. Thus increasing  the performance angle increases the performance of the lens at focusing light from higher incident angles.
    performance_angle = -1
    performance_index = -1
    for i in range(num):
        #angle = np.pi/4
        angle = angles[i]
        for ev in sim.simulate(photon_rays_angledx(10000.0, 1.0, angle), keep_photons_beg=True, keep_photons_end=True, run_daq=True, max_steps=1000):

            detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)
            nohit = (ev.photons_end.flags & (0x1 << 0)).astype(bool)
            surface_absorb = (ev.photons_end.flags & (0x1 << 3)).astype(bool)
            bulk_absorb = (ev.photons_end.flags & (0x1 << 1)).astype(bool)
            #print 'detected ' + str(np.sum(detected))
            print 'nohit ' + str(np.sum(nohit))
            #print 'surface absorb '+ str(np.sum(surface_absorb))
            #print 'bulk absorb ' + str(np.sum(bulk_absorb))
            #print 'total ' + str(np.sum(detected)+np.sum(nohit)+np.sum(surface_absorb)+np.sum(bulk_absorb))
            #f.write_event(ev) 

            radius_std = np.std(np.sqrt((ev.photons_end.pos[detected][:,0])**2 + (ev.photons_end.pos[detected][:,2])**2))
            deviations[i] = radius_std
            if performance_angle == -1:
                if radius_std >= 0.5:
                    performance_angle = angle
                    performance_index = i
        
            #print np.std(ev.photons_end.pos[detected][:,0]), np.std(ev.photons_end.pos[detected][:,2])
            #print radius_std 
    
    #preperfangle_sum adds up all the radius_std's up to but NOT including the performance_index.
    #postperfangle_sum adds up all the radius_std's after the performance index including radiusstd[performance_index]
    preperfangle_sum = 0
    postperfangle_sum = 0
    for i in range(num):
        if i < performance_index:
            preperfangle_sum += deviations[i]
        else:
            postperfangle_sum += deviations[i]
    total_sum = preperfangle_sum + postperfangle_sum

    print preperfangle_sum
    print postperfangle_sum
    print total_sum
    print performance_angle
    print performance_index
    

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(angles, deviations)
    plt.axis((0, 1.1, 0, 8))
    plt.text(0.25, 7, 'performance angle: ' + str(performance_angle))
    plt.text(0.25, 6.7, 'preperfangle_sum: ' + str(preperfangle_sum))
    plt.text(0.25, 6.4, 'postperfangle_sum: ' + str(postperfangle_sum))
    plt.text(0.25, 6.1, 'total_sum: ' + str(total_sum))
    plt.xlabel('incident angle')
    plt.ylabel('standard deviation of final position')
    plt.title('PC Incident Angle vs STD for pcradius: ' + str(pcrad))
    plt.show()
        

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # #ax = fig.add_subplot(111, projection='3d')
    # xs = ev.photons_end.pos[detected][:,0]
    # ys = ev.photons_end.pos[detected][:,1]
    # zs = ev.photons_end.pos[detected][:,2]
    # ax.scatter(xs, ys, zs)
    # ax.set_xlabel('X final position')
    # ax.set_ylabel('Y final position')
    # ax.set_zlabel('Z final position')
    # plt.show()
    

    # fig = plt.figure(figsize=(8, 8))
    # plt.hist(ev.photons_end.pos[detected][:,0],100)
    # plt.xlabel('x position (Ym)')
    # plt.ylabel('number of photons (GeV)')
    # plt.title('Photon Hit Locations- X')
    # plt.show()

    # fig = plt.figure(figsize=(8, 8))
    # plt.hist(ev.photons_end.pos[detected][:,2],100)
    # plt.xlabel('z position (Ym)')
    # plt.ylabel('number of photons (GeV)')
    # plt.title('Photon Hit Locations- Z')
    # plt.show()

    # fig = plt.figure(figsize=(8, 8))
    # plt.scatter(ev.photons_beg.pos[detected][:,0], ev.photons_end.pos[detected][:,0])
    # plt.xlabel('initial photon x-position (Ym)')
    # plt.ylabel('finial photon x-position (Ym)')
    # plt.title('Photon Initial vs Final X-Location')
    # plt.show()
        
    # fig = plt.figure(figsize=(8, 8))
    # plt.scatter(ev.photons_beg.pos[detected][:,2], ev.photons_end.pos[detected][:,2])
    # plt.xlabel('initial photon z-position (Ym)')
    # plt.ylabel('finial photon z-position (Ym)')
    # plt.title('Photon Initial vs Final Z-Location')
    # plt.show()
        
    # fig = plt.figure(figsize=(8, 8))
    # plt.scatter(ev.photons_beg.pos[detected][:,0], ev.photons_beg.pos[detected][:,2])
    # plt.xlabel('initial photon x-position (Ym)')
    # plt.ylabel('initial photon z-position (Ym)')
    # plt.title('Photon Initial Location Plot')
    # plt.show()

    # fig = plt.figure(figsize=(8, 8))
    # plt.scatter(ev.photons_end.pos[detected][:,0], ev.photons_end.pos[detected][:,2])
    # plt.xlabel('final photon x-position (Ym)')
    # plt.ylabel('final photon z-position (Ym)')
    # plt.title('Photon Final Location Plot')
    # plt.show()
        
    # fig = plt.figure(figsize=(8,8))
    # plt.hist2d(ev.photons_end.pos[detected][:,0], ev.photons_end.pos[detected][:,2], bins=1000)
    # plt.colorbar()
    # plt.show()
        
    # fig = plt.figure(figsize=(8, 8))
    # plt.scatter(np.sqrt((ev.photons_beg.pos[detected][:,0])**2+(ev.photons_beg.pos[detected][:,2])**2), np.sqrt((ev.photons_end.pos[detected][:,0])**2+(ev.photons_end.pos[detected][:,2])**2))
    # plt.xlabel('initial photon radius (Ym)')
    # plt.ylabel('final radius (Ym)')
    # plt.title('Photon Initial vs. Final Radius Plot')
    # plt.show()
    
    f.close()
