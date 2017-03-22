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
import meshhelper as mh
from scipy import optimize, ndimage, spatial
import lensmaterials as lm
import numpy as np

def asphere_func(x, c, k, d, e, f):
    return c*(x**2)/(1+np.sqrt(1-(1+k)*(c**2)*(x**2))) + d*(x**2) + e*(x**4) + f*(x**6)

def asphere_lens(rad, t, c1, k1, d1, e1, f1, c2, k2, d2, e2, f2, nsteps=64):
    # rad - radius of lens
    # t - thickness at widest point (if this is too small for the desired rad, will use min consistent thickness)
    # c - 1/(radius of curvature) in 1/mm
    # k - conic constant (unitless)
    # d - quadratic (1/mm); e - quartic (1/mm^3); f - sextic (1/mm^5)
    # First surface should be before second, so c1 > c2 must hold (for all signs of c1 and c2)
    # Note that parameters do not all scale by the same factor (not linear)
    # To multiply lens size by a, multiply diameter by a, divide c and d by a, divide e by a^3, etc.
    # Center at X=0, Y=0; rotated about x-axis, Y=0 plane is edge of first surface
    ymax_1 = asphere_func(rad, c1, k1, d1, e1, f1)
    ymax_2 = asphere_func(rad, c2, k2, d2, e2, f2)
    
    if t < ymax_1 - ymax_2: # Avoid situations where the two lens faces overlap each other
        #print ymax_1, ymax_2, ymax_1-ymax_2, t
        t = ymax_1 - ymax_2
    
    x1 = np.linspace(0., rad, nsteps/2) # Space out points linearly in x
    x2 = np.linspace(rad, 0., nsteps/2)
    
    y1 = asphere_func(x1, c1, k1, d1, e1, f1)-ymax_1 # Shift so Y=0 plane is the first surface's edge
    y2 = asphere_func(x2, c2, k2, d2, e2, f2)+t-ymax_1 # Shift second surface to give appropriate thickness
    
    #print x1, x2
    #print y1, y2
    
    # fig = plt.figure(figsize=(8, 8))
    # plt.scatter(np.concatenate((x1,x2)), np.concatenate((y1,y2)))
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Lens coords')
    # plt.show()
    
    return make.rotate_extrude(np.concatenate((x1, x2)), np.concatenate((y1, y2)), nsteps=256)#64)
    
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

   
def cylindrical_shell(inner_radius, outer_radius, height, nsteps=1024):
    if inner_radius < outer_radius:
        return make.rotate_extrude([inner_radius, outer_radius, outer_radius, inner_radius], [-height/2.0, -height/2.0, height/2.0, height/2.0], nsteps)
    else:
        print "inner radius must be less than outer radius"

def photon_rays_angledy(n, diameter, ypos, theta):
    #constructs collimated photons traveling at a specified angle from the y-axis in the xy-plane. z_initial=0, y_initial= ypos. Photons travel towards the origin. I'm not sure if this works for negative ypos, but it probably does. Theta can range from -pi/2 to pi/2
    pos = np.array([(i + ypos*np.tan(theta), ypos, 0) for i in np.linspace(-diameter/2.0, diameter/2.0, n)])
    dir = np.tile((-np.sin(theta), -np.cos(theta), 0), (n,1))
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000, n)
    return Photons(pos, dir, pol, wavelengths)

def photon_rays_angledy_circle(n, diameter, ypos, theta):
    #constructs collimated photons traveling in a uniform circle at a specified angle from the y-axis in the xz-plane at y-position ypos.
    #theta can range from -pi/2 to pi/2
    radii = np.random.uniform(0, diameter/2.0, n)
    angles = np.random.uniform(0, 2*np.pi, n)
    points = np.empty((n,3))
    points[:,0] = np.sqrt(diameter/2.0)*np.sqrt(radii)*np.cos(angles) + ypos*np.tan(theta)
    points[:,1] = np.repeat(ypos, n)
    points[:,2] = np.sqrt(diameter/2.0)*np.sqrt(radii)*np.sin(angles)
    pos = points
    dir = np.tile((-np.sin(theta), -np.cos(theta), 0), (n,1))
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000, n)
    return Photons(pos, dir, pol, wavelengths)
    
def perf_radius(target, coordarray):
    #finds the minimum radius from the centroid of a coordinate array that contains target amount of coordinates
    centroid = np.mean(coordarray, 0)
    tree = spatial.KDTree(coordarray)
    def F(rad):
        neighbors = tree.query_ball_point(centroid, rad, eps=1e-6)
        return (np.shape(neighbors)[0] - target)**2
    radius = optimize.fmin(F, 0.1)
    if abs(np.shape(tree.query_ball_point(centroid, radius, eps=1e-6))[0] - target) > 100:
        radius = optimize.fmin(F, 0.01)
        if abs(np.shape(tree.query_ball_point(centroid, radius, eps=1e-6))[0] - target) > 100:
            radius = optimize.fmin(F, 0.5)
    return radius

def find_main_deviation(radius, coordarray):
    #computes the standard deviation of the coordinates within 'radius' of the centroid of 'coordarray'. NOTE: standard deviation calculation assumes that y-pos is constant for coordarray.
    centroid = np.mean(coordarray, 0)
    redwood = spatial.KDTree(coordarray)
    neighbors = redwood.query_ball_point(centroid, radius, eps=1e-6)
    points = coordarray[neighbors]
    deviation = np.sqrt(np.add((np.std(points[:,0]))**2, (np.std(points[:,2]))**2))
    return deviation

# testing findR2:    
# R1 = 1
# focal_length = 0.5
# foundR2 = findR2(R1, 1.0, focal_length)
# print foundR2
# print spherical_focal_length(R1, -1.1, 1.0)
 
if __name__ == '__main__':
    from chroma.sim import Simulation
    from chroma import sample
    from chroma.event import Photons
    from chroma.loader import load_bvh
    from chroma.generator import vertex
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    
    shiftin = 0.# 10.0
    focal_length = 1074. # From center of lens to detector; 660. from surface 2 to detector
    
    # asphere lens
    as_rad = 488.
    diameter = 2.0*as_rad
    as_t = 506.981
    
    as_c1 = 1./820.77
    as_k1 = -7.108
    as_d1 = 2.028e-4
    as_e1 = -1.294e-9
    as_f1 = 1.152e-15
    
    as_c2 = -1./487.388
    as_k2 = -0.078
    as_d2 = -2.412e-4
    as_e2 = 9.869e-10
    as_f2 = -1.49e-15
    
    as_mesh = mh.rotate(asphere_lens(as_rad, as_t, as_c1, as_k1, as_d1, as_e1, as_f1, as_c2, as_k2, as_d2, as_e2, as_f2, nsteps=256), make_rotation_matrix(-np.pi, (1,0,0)))
    as_solid = Solid(as_mesh, lm.lensmat, lm.ls)

    # Photodetector
    det_rad = 643.
    det_r_curve = 943.
    #pmt = Solid(make.box(det_rad*2,10.0,det_rad*2), glass, lm.ls, surface=lm.fulldetect, color=0x00ff00)
    pmt_mesh = mh.rotate(curved_surface(det_r_curve, diameter=2*det_rad, nsteps=64), make_rotation_matrix(np.pi, (1,0,0)))
    pmt = Solid(mh.shift(pmt_mesh, (0,0,0)), lm.ls, lm.ls, lm.fulldetect, 0x0000FF)

    # photon initial loc
    photon_start_y = focal_length*0.2

    # blocker
    EPD_ratio = 0.7
    blocker = Solid(cylindrical_shell(as_rad*EPD_ratio, 1000.0, 1.0), glass, lm.ls, black_surface, 0xff0000)
    #blocker = Solid(cylindrical_shell(1.0, 1000.0, 1.0), glass, lm.ls, black_surface, 0xff0000)
    testobject = Solid(make.box(10.0, 10.0, 10.0), glass, lm.ls, color=0x00ff00)
    det_blocker = Solid(make.box(det_rad*4, 10.0, det_rad*4), glass, lm.ls, black_surface, 0xff0000)
    #ultraabsorb = Solid(make.box(100, 100, 100), lm.ls, lm.ls, surface = lm.fulldetect)

    dummy_t = 10.
    dummy_lens = Solid(make.box(as_rad*3, dummy_t, as_rad*3), lm.lensmat, lm.ls, color=0x00ff00)

    #creates the entire Detector and adds the solid & pmt
    ftlneutrino = Detector(lm.ls)
    #ftlneutrino.add_solid(ultraabsorb)
    #ftlneutrino.add_solid(lens_solid2)
    ftlneutrino.add_solid(as_solid)
    ftlneutrino.add_solid(blocker)
    ftlneutrino.add_solid(det_blocker, displacement=(0,focal_length*0.5, 0))
    #ftlneutrino.add_solid(dummy_lens, rotation=make_rotation_matrix(1.2*np.pi/3., (1,0,0)), displacement=(0, -dummy_t/2.0-400., 0))
    # Edge of lens at -412.0
    #ftlneutrino.add_solid(testobject, rotation=None, displacement=(0, photon_start_y, 0))
    #ftlneutrino.add_pmt(pmt, rotation=None, displacement=(0,-0.5-3.0, 0))
    ftlneutrino.add_pmt(pmt, rotation=None, displacement=(0,-focal_length + shiftin, 0))
    #ftlneutrino.add_pmt(pmt, rotation=None, displacement=(0,-0.5-pcfocal_length + shiftin, 0))

    ftlneutrino.flatten()
    ftlneutrino.bvh = load_bvh(ftlneutrino)
    
    # from chroma.io.root import RootWriter
    # f = RootWriter('test.root')

    view(ftlneutrino)
    # quit()

    #performance_angle represents the last angle whose deviation is less than 0.5. Thus increasing  the performance angle increases the performance of the lens at focusing light from higher incident angles. goal*detn = target amount of photons for the perf_radius function.  (detn is the amount of detected photons.)
    num_photons = 10000.
    num_angles = 6
    max_angle = 0.873 # 50 degrees, in rad
    angles = np.linspace(0., max_angle, num_angles)
    print 'angles: '+str(angles)
    deviations = np.zeros(num_angles)
    perf_radii = np.zeros(num_angles)
    goal = 0.95
    #std of initial photons == perf_indicator
    perf_indicator = np.sqrt(((goal*diameter/2.0)**2)/3.0)
    perf_rad_indicator = diameter/2.0*goal
    perf_rad_angle = -1
    performance_angle = -1
    performance_index = -1
    xstd = 0
    radstd = 0

    sim = Simulation(ftlneutrino)
    for i in range(num_angles):
        #angle = np.radians(45)
        angle = angles[i]
        for ev in sim.simulate(photon_rays_angledy_circle(num_photons, diameter, photon_start_y, angle), keep_photons_beg=True, keep_photons_end=True, run_daq=True, max_steps=1000):

            detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)
            detn = np.sum(detected)
            nohit = (ev.photons_end.flags & (0x1 << 0)).astype(bool)
            surface_absorb = (ev.photons_end.flags & (0x1 << 3)).astype(bool)
            bulk_absorb = (ev.photons_end.flags & (0x1 << 1)).astype(bool)
            refl_diff = (ev.photons_end.flags & (0x1 << 5)).astype(bool)
            refl_spec = (ev.photons_end.flags & (0x1 << 6)).astype(bool)
            detect_refl = refl_spec & detected
            print 'reflected and detected '+str(np.sum(detect_refl))
            print 'detected ' + str(detn)
            print 'nohit ' + str(np.sum(nohit))
            print 'surface absorb '+ str(np.sum(surface_absorb))
            print 'reflect diffuse '+ str(np.sum(refl_diff))
            print 'reflect specular '+ str(np.sum(refl_spec))
            print 'bulk absorb ' + str(np.sum(bulk_absorb))
            print 'total ' + str(np.sum(detected)+np.sum(nohit)+np.sum(surface_absorb)+np.sum(bulk_absorb))
             
            startpos = ev.photons_beg.pos[detected]
            radii0 = np.sqrt(np.add(startpos[:,0]**2, startpos[:,2]**2))


            endpos = ev.photons_end.pos[detected]
            
            perf_rad = perf_radius(goal*detn, endpos)
            deviation = find_main_deviation(perf_rad, endpos)
            radii = np.sqrt(np.add(endpos[:,0]**2, endpos[:,2]**2))
            radstd = np.std(radii)
            deviations[i] = deviation
            if deviation <= perf_indicator:
               performance_angle = angle
               performance_index = i

            # fig = plt.figure(figsize=(7.8, 6))
            # plt.hist(radii0, bins=np.linspace(-2, det_rad*2, 150))
            # #plt.axis((-2.0, 2.0, 0, 3000))
            # plt.text(-1.5, 2500, 'standard deviation of radius: ' + str(deviation))
            # plt.xlabel('Initial Radius (mm)')
            # plt.ylabel('Amount of Photons')
            # plt.title('Start Locations')
            # plt.show()

            # fig = plt.figure(figsize=(7.8, 6))
            # plt.hist(radii, bins=np.linspace(-2, det_rad*2, 150))
            # #plt.axis((-2.0, 2.0, 0, 3000))
            # plt.text(-1.5, 2500, 'standard deviation of radius: ' + str(deviation))
            # plt.xlabel('Final Radius (mm)')
            # plt.ylabel('Amount of Photons')
            # plt.title('Hit Detection Locations')
            # plt.show()

            fig = plt.figure(figsize=(7.8, 6))
            plt.hist2d(startpos[:,0], startpos[:,2], bins=100)
            plt.axis((-det_rad, det_rad, -det_rad, det_rad))
            plt.xlabel('Initial X-Position (mm)')
            plt.ylabel('Initial Z-Position (mm)')
            plt.title('Start Locations')
            plt.colorbar()
            plt.show()

            fig = plt.figure(figsize=(7.8, 6))
            plt.hist2d(endpos[:,0], endpos[:,2], bins=100)
            plt.axis((-det_rad, det_rad, -det_rad, det_rad))
            plt.xlabel('Final X-Position (mm)')
            plt.ylabel('Final Z-Position (mm)')
            plt.title('Hit Detection Locations')
            plt.colorbar()
            plt.show()

            # fig = plt.figure(figsize=(7.8, 6))
            # plt.hist(endpos[:,0], bins=np.linspace(-2, det_rad*2, 150))
            # #plt.axis((-2.0, 2.0, 0, 3000))
            # plt.text(-1.5, 2500, 'standard deviation of xpos: ' + str(xstd))
            # plt.xlabel('Final X-Position (mm)')
            # plt.ylabel('Amount of Photons')
            # plt.title('Hit Detection Locations')
            # plt.show()
            
            xstd = np.std(endpos[:,0])
             
            perf_radii[i] = perf_rad
            if perf_rad <= perf_rad_indicator:
                perf_rad_angle = angle
            
            #f.write_event(ev)
    
    print 'Done'

    # radpre45_sum = 0
    # radpost45_sum = 0
    # for i in range(num_angles):
    #     if angles[i] < np.pi/4:
    #         radpre45_sum += perf_radii[i]
    #     else:
    #         radpost45_sum += perf_radii[i]
    # radtotal_sum = radpre45_sum + radpost45_sum

    # pre45_sum = 0
    # post45_sum = 0
    # for i in range(num_angles):
    #     if angles[i] < np.pi/4:
    #         pre45_sum += deviations[i]
    #     else:
    #         post45_sum += deviations[i]
    # total_sum = pre45_sum + post45_sum

    # # print 'performance_index: ' + str(performance_index)
    
    # if shiftin == 0:
    #     shift_string = ''
    # elif shiftin > 0:
    #     shift_string = ' with pmt shift-in: ' + str(shiftin)
    # else:
    #     shift_string = ' with pmt shift-out: ' + str(shiftin)
    
    # #plot for spherical lenses:
    # fig = plt.figure(figsize=(8, 8))
    # plt.scatter(angles, deviations)
    # plt.axis((0, 1.1, 0, 3))
    # plt.text(0.25, 2.9, 'focal length: ' + str(focal_length))
    # plt.text(0.25, 2.8, 'goal: ' + str(goal))
    # plt.text(0.25, 2.7, 'performance indicator: ' + str(perf_indicator))
    # plt.text(0.25, 2.6, 'performance angle: ' + str(performance_angle))
    # plt.text(0.25, 2.5, 'pre45_sum: ' + str(pre45_sum))
    # plt.text(0.25, 2.4, 'post45_sum: ' + str(post45_sum))
    # plt.text(0.25, 2.3, 'total_sum: ' + str(total_sum))
    # plt.xlabel('Incident Angle')
    # plt.ylabel('Deviation')
    # plt.title('Incident Angle vs Deviation for R1: ' + str(R1) + ' R2 ' + str(R2) + ' and focal_length: ' + str(focal_length) + shift_string)
    # plt.show()
        
    # fig = plt.figure(figsize=(8, 8))
    # plt.scatter(angles, perf_radii)
    # plt.axis((0, 1.1, 0, 3))
    # plt.text(0.25, 2.9, 'focal length: ' + str(focal_length))
    # plt.text(0.25, 2.8, 'goal: ' + str(goal))
    # plt.text(0.25, 2.7, 'perf_rad_indicator: ' + str(perf_rad_indicator))
    # plt.text(0.25, 2.6, 'perf_rad_angle: ' + str(perf_rad_angle))
    # plt.text(0.25, 2.5, 'radpre45_sum: ' + str(radpre45_sum))
    # plt.text(0.25, 2.4, 'radpost45_sum: ' + str(radpost45_sum))
    # plt.text(0.25, 2.3, 'radtotal_sum: ' + str(radtotal_sum))
    # plt.xlabel('Incident Angle')
    # plt.ylabel('Performance Radius')
    # plt.title('Incident Angle vs perf_radius for R1: ' + str(R1) + ' R2 ' + str(R2) + ' and focal_length: ' + str(focal_length) + shift_string)
    # plt.show()

    #plots for pc lenses:
    # fig = plt.figure(figsize=(8, 8))
    # plt.scatter(angles, deviations)
    # plt.axis((0, 1.1, 0, 8))
    # plt.text(0.25, 7.3, 'pcfocal_length: ' + str(pcfocal_length))
    # plt.text(0.25, 7, 'performance indicator: ' + str(perf_indicator))
    # plt.text(0.25, 6.7, 'performance angle: ' + str(performance_angle))
    # plt.text(0.25, 6.4, 'pre45_sum: ' + str(pre45_sum))
    # plt.text(0.25, 6.1, 'post45_sum: ' + str(post45_sum))
    # plt.text(0.25, 5.8, 'total_sum: ' + str(total_sum))
    # plt.xlabel('Incident Angle')
    # plt.ylabel('Standard Deviation of Final Position')
    # plt.title('PC Incident Angle vs STD for pcradius: ' + str(pcrad) + shift_string)
    # plt.show()

    # fig = plt.figure(figsize=(8, 8))
    # plt.scatter(angles, perf_radii)
    # plt.axis((0, 1.1, 0, 5))
    # plt.text(0.25, 4.7, 'pcfocal_length: ' + str(pcfocal_length))
    # plt.text(0.25, 4.5, 'goal: ' + str(goal))
    # plt.text(0.25, 4.3, 'perf_rad_indicator: ' + str(perf_rad_indicator))
    # plt.text(0.25, 4.1, 'perf_rad angle: ' + str(perf_rad_angle))
    # plt.text(0.25, 3.9, 'pre45_sum: ' + str(radpre45_sum))
    # plt.text(0.25, 3.7, 'post45_sum: ' + str(radpost45_sum))
    # plt.text(0.25, 3.5, 'total_sum: ' + str(radtotal_sum))
    # plt.xlabel('Incident Angle')
    # plt.ylabel('Performance Radius')
    # plt.title('PC Incident Angle vs perf_radius for pcradius: ' + str(pcrad) + shift_string)
    # plt.show()
        
    #other plots:
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

    # fig = plt.figure(figsize=(7.8, 6))
    # plt.hist2d(ev.photons_end.pos[detected][:,0], ev.photons_end.pos[detected][:,2], bins=100)
    # #plt.axis((-2.0, 2.0, -0.5, 0.5))
    # plt.xlabel('Final X-Position (mm)')
    # plt.ylabel('Final Z-Position (mm)')
    # plt.title('Hit Detection Locations')
    # plt.colorbar()
    # plt.show()

    # fig = plt.figure(figsize=(7.8, 6))
    # plt.scatter(ev.photons_end.pos[detected][:,0], ev.photons_end.pos[detected][:,2])
    # plt.axis((-2.0, 2.0, -0.5, 0.5))
    # plt.xlabel('Final X-Position (mm)')
    # plt.ylabel('Final Z-Position (mm)')
    # plt.title('Hit Detection Locations')
    # plt.show()
    
    #f.close()
