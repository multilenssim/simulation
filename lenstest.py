from chroma import make, view
from chroma.geometry import Geometry, Material, Solid, Surface
from chroma.transform import make_rotation_matrix, normalize
from chroma.demo.optics import glass, water, vacuum
from chroma.demo.optics import black_surface, r7081hqe_photocathode
from chroma.loader import create_geometry_from_obj
from chroma.detector import Detector
from chroma.pmt import build_pmt
from chroma.event import Photons
from chroma.sim import Simulation
from chroma.sample import uniform_sphere
from scipy import optimize, ndimage, spatial
import lensmaterials as lm
import numpy as np

def pclens(radius, diameter, nsteps=1024):
    #NOTE, I AM USING pclens2 BELOW, NOT THIS ONE. BOTH BUILD THE MESH IN DIFFERENT WAYS
    #constructs a plano-convex lens mesh
    angles = np.linspace(-np.pi/2, np.pi/2, nsteps)
    return make.rotate_extrude(diameter/2.0*np.cos(angles), (0.5*(np.sign(angles)+1))*(np.sqrt(radius**2-(diameter/2*np.cos(angles))**2)-np.sqrt(radius**2-(diameter/2)**2)), nsteps=64)

def pclens2(radius, diameter, nsteps=512):
    #works best with angles linspace endpoint=True
    halfd = diameter/2.0
    shift = np.sqrt(radius**2-(halfd)**2)
    theta = np.arctan(shift/halfd)
    angles = np.linspace(theta, np.pi/2, nsteps)
    x2 = radius*np.cos(angles)
    y2 = radius*np.sin(angles) - shift
    xs = np.concatenate(([0.0], x2))
    ys = np.concatenate(([0.0], y2))
    return make.rotate_extrude(xs, ys, nsteps=64)

def spherical_lens(R1, R2, diameter, nsteps=1024):
    '''constructs a spherical lens with specified radii of curvature. Works with meniscus lenses. If meniscus, light should go in the larger cap first for better performance- see MOE 4th ed. p.86. Make sure not to fold R1 through R2 or vica-versa in order to keep rotate_extrude going counterclockwise.
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
    return make.rotate_extrude(np.concatenate((x2,x1)), np.concatenate((y2,y1)), nsteps=64)
    
def cylindrical_shell(inner_radius, outer_radius, height, nsteps=1024):
    if inner_radius < outer_radius:
        return make.rotate_extrude([inner_radius, outer_radius, outer_radius, inner_radius], [-height/2.0, -height/2.0, height/2.0, height/2.0], nsteps)
    else:
        print "inner radius must be less than outer radius"

def single_ray(position, direction, num):
    pos = np.tile(position, (num, 1))
    dir = np.tile(direction, (num, 1))
    return Photons(pos, dir, np.tile((1/np.sqrt(2), 0, 1/np.sqrt(2)), (num ,1)), np.repeat(500, num))

def photon_rays_angledy(n, diameter, ypos, theta):
    #constructs collimated photons traveling at a specified angle from the y-axis in the xy-plane. z_initial=0, y_initial= ypos. Photons travel towards the origin. I'm not sure if this works for negative ypos, but it probably does. Theta can range from -pi/2 to pi/2
    pos = np.array([(i + ypos*np.tan(theta), ypos, 0) for i in np.linspace(-diameter/2.0, diameter/2.0, n)])
    dir = np.tile((-np.sin(theta), -np.cos(theta), 0), (n,1))
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000, n)
    return Photons(pos, dir, pol, wavelengths)

def pc_focal_length(radius):
    #lensmaker's equation where R2 -> infinity
    n1 = float(lm.ls_refractive_index)
    n2 = float(lm.lensmat_refractive_index)
    focal_length = 1.0/((n2/n1-1.0)*(1.0/radius))
    return focal_length

def spherical_focal_length(R1, R2, diameter):
    #lensmaker's equation
    shift1 = -np.sign(R1)*np.sqrt(R1**2 - (diameter/2.0)**2)
    shift2 = -np.sign(R2)*np.sqrt(R2**2 - (diameter/2.0)**2)
    thickness = abs((R1+shift1)-(R2+shift2))
    #print 'thickness: ' + str(thickness)
    #index = float(lm.lensmat_refractive_index)/lm.ls_refractive_index
    n2 = float(lm.lensmat_refractive_index)
    n1 = float(lm.ls_refractive_index)
    focal_length = 1.0/((n2/n1-1)*(1.0/R1 - 1.0/R2 + thickness*(n2-n1)/(n2*R1*R2)))
    return focal_length

def perf_radius(target, coordarray):
    #finds the minimum radius from the centroid of a coordinate array that contains target amount of coordinates
    centroid = np.mean(coordarray, 0)
    tree = spatial.KDTree(coordarray)
    def F(rad):
        neighbors = tree.query_ball_point(centroid, rad, eps=1e-6)
        return (np.shape(neighbors)[0] - target)**2
    radius = optimize.fmin(F, 0.1)
    if abs(np.shape(tree.query_ball_point(centroid, radius, eps=1e-6))[0] - target) > 100:
        radius = optimize.fmin(F, 1.0)
    return radius

def find_main_deviation(radius, coordarray):
    #computes the standard deviation of the coordinates within 'radius' of the centroid of 'coordarray'. NOTE: standard deviation calculation assumes that y-pos is constant for coordarray.
    centroid = np.mean(coordarray, 0)
    redwood = spatial.KDTree(coordarray)
    neighbors = redwood.query_ball_point(centroid, radius, eps=1e-6)
    points = coordarray[neighbors]
    deviation = np.sqrt(np.add((np.std(points[:,0]))**2, (np.std(points[:,2]))**2))
    return deviation
 
if __name__ == '__main__':
    from chroma.sim import Simulation
    from chroma import sample
    from chroma.event import Photons
    from chroma.loader import load_bvh
    from chroma.generator import vertex
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    def snells_law_test(initial_angle):
        #tests snell's law for a flat surface.
        ni = lm.ls_refractive_index
        nf = lm.lensmat_refractive_index
        
        flatlens = Solid(make.box(100, 10, 100), lm.lensmat, lm.ls, color=0xff0000)
        flatlenstestpmt = Solid(make.box(99, 1.0, 99), lm.lensmat, lm.lensmat, surface=lm.fulldetect, color=0x00ff00)
        testobject = Solid(make.box(0.5, 0.5, 0.5), glass, lm.ls, color=0xff0000)
        
        ftlneutrino = Detector(lm.ls)
        ftlneutrino.add_solid(flatlens, displacement=(0, -5, 0))
        ftlneutrino.add_pmt(flatlenstestpmt, displacement=(0, -5.5, 0))
        # ftlneutrino.add_solid(testobject)
        # ftlneutrino.add_solid(testobject, displacement=(0, -5, 0))
        
        ftlneutrino.flatten()
        ftlneutrino.bvh = load_bvh(ftlneutrino)
        
        # view(ftlneutrino)
        # quit()
        
        sim = Simulation(ftlneutrino)

        for ev in sim.simulate(photon_rays_angledy(10000.0, 1.0, 5.0, initial_angle), keep_photons_beg=True, keep_photons_end=True, run_daq=True, max_steps=1000):
        
            detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)
            detn = np.sum(detected)
            print 'detected: ' + str(detn)
            
            begpos = ev.photons_beg.pos[detected]
            endpos = ev.photons_end.pos[detected]
            begdir = ev.photons_beg.dir[detected]
            enddir = ev.photons_end.dir[detected]
            
            print 'begpos:' + str(begpos)
            print 'endpos:' + str(endpos)
            print 'begdir:' + str(begdir)
            print 'enddir:' + str(enddir)
            
            predicted_final_angle = np.arcsin(ni/nf*np.sin(initial_angle))
            final_angles = np.arccos(np.dot(enddir, (0, -1, 0)))
            
            #checks: (should see each thing printed twice)
            print 'CHECKS:'
            print 'beginning position:'
            print begpos[detn/2,0]
            print 5*np.tan(initial_angle)
            print 'initial angle:'
            print initial_angle
            print np.arccos(np.dot(begdir, (0, -1, 0)))
            print 'ending position:'
            print endpos[detn/2,0]
            print -5*np.tan(final_angles)

        return 'predicted final angle: ' + str(predicted_final_angle) + ' actual final angle: ' + str(final_angles)
        
    #print snells_law_test(np.pi/3)    

    #Specifications of the lens parameters are made here as global variables since I am currently testing many things that use these same parameters.

    diameter = 1.0
    #std of initial photons == perf_indicator
    perf_indicator = np.sqrt(((diameter/2.0)**2)/3.0)

    R1 = 1
    R2 = -1
    focal_length = spherical_focal_length(R1, R2, diameter)
    
    pcrad = 3.0
    pcfocal_length = pc_focal_length(pcrad)

    #defining the specifications of the meshes
    lens2 = pclens2(pcrad, diameter)
    lensm = spherical_lens(R1, R2, diameter)
    
    #creating solids of form (mesh, inside_mat, outside_mat)
    lens_solid2 = Solid(lens2, lm.lensmat, lm.ls)
    lens_solidm = Solid(lensm, lm.lensmat, lm.ls)

    pmt = Solid(make.box(100.0,1.0,100.0), lm.lensmat, lm.ls, surface=lm.fulldetect, color=0x00ff00)
    blocker = Solid(cylindrical_shell(diameter/2.0, 100.0, 0.00001), lm.lensmat, lm.ls, black_surface, 0xff0000)
    testobject = Solid(make.box(0.01, 0.01, 0.01), glass, lm.ls, color=0xff0000)
      
    def best_pmt_position(angle, num, goal, shape):
        #use this to find the best pmt position
        #use shape 'pc' or 'spherical'
        #The specifications of the lenses (such as radii of curvature and diameter) are made global variables earlier because it is useful to do so currently when testing other aspects.
        perf_rad_indicator = diameter/2.0*goal
        perf_radii = -np.ones(num)
        deviations = -np.ones(num)
        if shape == 'pc':
            pmt_positions = np.linspace(0.01, pcfocal_length, num)
            #pmt_positions = np.linspace(0.5*pcfocal_length, 1.5*pcfocal_length, num)
            print 'pcfocal_length: ' + str(pcfocal_length)
        elif shape == 'spherical':
            pmt_positions = np.linspace(0.01, focal_length, num)
            #pmt_positions = np.linspace(0.5*focal_length, 1.5*focal_length, num, endpoint=True)
            print 'focal_length: ' + str(focal_length)
        else:
            print 'need shape'

        for i in range(num):
            deviations[i], perf_radii[i] = find_perf_radii(angle, pmt_positions[i], goal, shape)
            print i
        lowest_deviation_index = np.argmin(deviations)
        lowest_perf_radius_index = np.argmin(perf_radii)
        deviation_pmt_pos = pmt_positions[lowest_deviation_index]
        perf_radius_pmt_pos = pmt_positions[lowest_perf_radius_index]

        low_bound = pmt_positions[0]
        high_bound = pmt_positions[num-1]
        mid = pmt_positions[int(num)/4]
        if shape == 'pc':
            fig = plt.figure(figsize=(8, 8))
            plt.scatter(pmt_positions, deviations)
            plt.axis((low_bound, high_bound, 0, 0.5))
            plt.text(mid, 0.45, 'pcfocal_length: ' + str(pcfocal_length))
            plt.text(mid, 0.43, 'perf_indicator: ' + str(perf_indicator))
            plt.xlabel('PMT Position')
            plt.ylabel('Standard Deviation of Final Position')
            plt.title('PC PMT Position vs STD for pcradius: ' + str(pcrad) + ' at angle: ' + str(angle))
            plt.show()

            fig = plt.figure(figsize=(8, 8))
            plt.scatter(pmt_positions, perf_radii)
            plt.axis((low_bound, high_bound, 0, 0.5))
            plt.text(mid, 0.45, 'pcfocal_length: ' + str(pcfocal_length))
            plt.text(mid, 0.43, 'goal: ' + str(goal))
            plt.text(mid, 0.41, 'perf_rad_indicator: ' + str(perf_rad_indicator))
            plt.xlabel('PMT Position')
            plt.ylabel('Performance Radius of Final Position')
            plt.title('PC PMT Position vs Performance Radii for pcradius: ' + str(pcrad) + ' at angle: ' + str(angle))
            plt.show()

        elif shape == 'spherical':
            fig = plt.figure(figsize=(8, 8))
            plt.scatter(pmt_positions, deviations)
            plt.axis((low_bound, high_bound, 0, 0.5))
            plt.text(mid, 0.45, 'focal_length: ' + str(focal_length))
            plt.text(mid, 0.43, 'perf_indicator: ' + str(perf_indicator))
            plt.xlabel('PMT Position')
            plt.ylabel('Standard Deviation of Final Position')
            plt.title('PMT Position vs STD for R1: ' + str(R1) + ' and R2: ' + str(R2) + ' at angle: ' + str(angle))
            plt.show()

            fig = plt.figure(figsize=(8, 8))
            plt.scatter(pmt_positions, perf_radii)
            plt.axis((low_bound, high_bound, 0, 0.5))
            plt.text(mid, 0.45, 'focal_length: ' + str(focal_length))
            plt.text(mid, 0.43, 'goal: ' + str(goal))
            plt.text(mid, 0.41, 'perf_rad_indicator: ' + str(perf_rad_indicator))
            plt.xlabel('PMT Position')
            plt.ylabel('Performance Radius of Final Position')
            plt.title('PMT Position vs Performance Radii for R1: ' + str(R1) + ' R2: ' + str(R2) + ' at angle: ' + str(angle))
            plt.show()

        return deviation_pmt_pos, perf_radius_pmt_pos

    def find_perf_radii(angle, pmtpos, goal, shape):
        #uses the pmt position and finds the perf_radii (and deviation) for the given angle. perf-radius (short for performance radius) is the radius that encapsulates (goal)*(number of detected photons) in the final positions of the photons.

        #creates the entire Detector and adds the solid & pmt
        ftlneutrino = Detector(lm.ls)
        if shape == 'pc':
            ftlneutrino.add_solid(lens_solid2)
        elif shape == 'spherical':
            ftlneutrino.add_solid(lens_solidm)
        else:
            print "need shape"

        ftlneutrino.add_solid(blocker)
        #ftlneutrino.add_solid(testobject, rotation=None, displacement=(0,0.152, 0))
        ftlneutrino.add_pmt(pmt, rotation=None, displacement=(0,-0.5 - pmtpos, 0))

        ftlneutrino.flatten()
        ftlneutrino.bvh = load_bvh(ftlneutrino)
        
        # view(ftlneutrino)
        # quit()

        sim = Simulation(ftlneutrino)

        for ev in sim.simulate(photon_rays_angledy(10000.0, diameter, 10.0, angle), keep_photons_beg=True, keep_photons_end=True, run_daq=True, max_steps=1000):

            detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)
            detn = np.sum(detected)
            print 'detected: ' + str(detn)
      
            endpos = ev.photons_end.pos[detected]

            perf_rad = perf_radius(goal*detn, endpos)
            #deviation = np.sqrt(np.add((np.std(endpos[:,0]))**2, (np.std(endpos[:,2]))**2))
            deviation = find_main_deviation(perf_rad, endpos)
            print 'pmtpos: ' + str(pmtpos) + ' perf_rad: ' + str(perf_rad) + ' deviation: ' + str(deviation)
        return deviation, perf_rad

print best_pmt_position(0, 8, 0.8, 'spherical')

  
