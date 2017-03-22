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
import meshhelper as mh
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

def pclens2(radius, diameter, nsteps=112):
    #works best with angles endpoint=True
    halfd = diameter/2.0
    shift = np.sqrt(radius**2-(halfd)**2)
    theta = np.arctan(shift/halfd)
    angles = np.linspace(theta, np.pi/2, nsteps)
    x2 = radius*np.cos(angles)
    y2 = radius*np.sin(angles) - shift
    xs = np.concatenate(([0.0], x2))
    ys = np.concatenate(([0.0], y2))
    return make.rotate_extrude(xs, ys, nsteps=256)
    
def pclens3(radius, diameter, nsteps=512):
    halfd = diameter/2.0
    shift = np.sqrt(radius**2-(halfd)**2)
    theta = np.arctan(shift/halfd)
    angles = np.linspace(np.pi/2, theta, nsteps)
    x2 = radius*np.cos(angles)
    y2 = radius*np.sin(angles) - shift
    xs = np.concatenate(([0.0], x2))
    ys = np.concatenate(([0.0], y2))
    return make.rotate_extrude(xs, ys, nsteps=64)

def cylindrical_pclens(radius, diameter, height, nsteps=512):
    shift = np.sqrt(radius**2-(diameter/2.0)**2)
    theta = np.arccos(diameter/(2.0*radius))
    angles = np.linspace(theta, np.pi-theta, nsteps)
    x1 = np.linspace(-diameter/2.0, diameter/2.0, nsteps/2)
    y1 = np.zeros(nsteps/2)
    x2 = radius*np.cos(angles)
    y2 = radius*np.sin(angles) - shift
    xs = np.concatenate((x1, x2))
    ys = np.concatenate((y1, y2))
    return make.linear_extrude(xs, ys, height)
    
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

def spherical_lens(R1, R2, diameter, nsteps=112):
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
    return make.rotate_extrude(np.concatenate((x2,x1)), np.concatenate((y2,y1)), nsteps=112)

def disk(radius, nsteps=32):
    return make.rotate_extrude([0, radius], [0, 0], nsteps)
    
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

def photon_rays_angledz(n, diameter, theta):
    #constructs collimated photons traveling at a specified angle from the y axis in the yz plane. y-initial=1
    #theta ranges from -pi/2 to pi/2
    pos = np.array([(i, 1, np.tan(theta)) for i in np.linspace(-diameter/2, diameter/2, n)])
    dir = np.tile((0,-1,-np.tan(theta)), (n,1))
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

def findR2(R1, diameter, focal_length):
    #finds the R2 necessary for a given focal length.
    #starts with an initial guess that is negative (and should hopefully work regardless of the value of R1 as long as R2 is indeed negative) and if it doesn't work chooses a value of R2 that is positive for positive R1. This is because having the wrong sign in the inital guess often messes it up.
    def F(findme):
        if (abs(R1) < diameter/2.0) or (abs(findme) < diameter/2.0):
            return 100
        else:
            return (spherical_focal_length(R1, findme, diameter) - focal_length)**2

    R2 = optimize.fmin(F, -0.5, xtol=1e-5)
    if ((spherical_focal_length(R1, R2, diameter) - focal_length)**2) > 1e-4:
        if R1 > 0:
            R2 = optimize.fmin(F, 2*R1, xtol=1e-5)
        else:
            R2 = optimize.fmin(F, -1, xtol=1e-5)
    return float(R2)

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

    diameter = 1.0
    #std of initial photons == perf_indicator
    perf_indicator = np.sqrt(((diameter/2.0)**2)/3.0)

    R1 = 1.426
    R2 = 2.384
    focal_length = spherical_focal_length(R1, R2, diameter)

    #focal_length = 6.0
    #R2 = findR2(R1, diameter, focal_length)
    
    pcrad = 5.4
    pcfocal_length = pc_focal_length(pcrad)

    #defining the specifications of the meshes
    lens1 = lens(5.0, diameter)
    lens2 = pclens2(pcrad, diameter)
    lens3 = paralens(0.5, diameter)
    lens4 = pcparalens(0.5, diameter)
    lens5 = make.box(1.0,1.0,1.0)
    lensm = spherical_lens(R1, R2, diameter)
    lensd = disk(diameter/2.0)
    lensc = cylindrical_pclens(pcrad, diameter, diameter)
    
    #creating solids of form (mesh, inside_mat, outside_mat)
    lens_solid1 = Solid(lens1, lm.lensmat, lm.ls)
    lens_solid2 = Solid(lens2, lm.lensmat, lm.ls)
    #lens_solid2 = Solid(lens2, lm.lensmat, lm.ls, lm.noreflect)
    lens_solid3 = Solid(lens3, lm.lensmat, lm.ls)
    lens_solid4 = Solid(lens4, lm.lensmat, lm.ls)
    lens_solid5 = Solid(lens5, lm.lensmat, lm.ls)
    lens_solidm = Solid(lensm, lm.lensmat, lm.ls)
    lens_solidd = Solid(lensd, lm.lensmat, lm.ls, surface=lm.fulldetect)
    lens_solidc = Solid(lensc, lm.lensmat, lm.ls)

    pmt = Solid(make.box(100.0, 1.0, 100.0), lm.lensmat, lm.lensmat, surface=lm.fulldetect, color=0x00ff00)
    smallpmt = Solid(make.box(2.0, 1.0, 2.0), lm.lensmat, lm.lensmat, surface=lm.fulldetect, color=0x00ff00)
    blocker = Solid(cylindrical_shell(diameter/2.0, 100.0, 0.00001), lm.lensmat, lm.ls, black_surface, 0xff0000)
    
    testobject = Solid(make.box(0.1, 0.1, 0.1), glass, lm.ls, color=0xff0000)
    #ultraabsorb = Solid(make.box(100, 100, 100), lm.ls, lm.ls, surface = lm.fulldetect)
    
    def find_perf_radii(angle, pmtpos, goal, shape):
        #uses the pmt position and finds the perf_radii (and deviation) for the given angle

        #creates the entire Detector and adds the solid & pmt
        ftlneutrino = Detector(lm.ls)
        if shape == 'pc':
            ftlneutrino.add_solid(lens_solid2)
            #ftlneutrino.add_solid(lens_solidc)
        elif shape == 'spherical':
            ftlneutrino.add_solid(lens_solidm)
            #ftlneutrino.add_solid(lens_solidm, rotation=make_rotation_matrix(np.pi/2.0, (0,1,0)), displacement=(0,0,0))
        else:
            print "need shape"

        ftlneutrino.add_solid(blocker)
        #ftlneutrino.add_solid(testobject, rotation=None, displacement=(0, -focal_length, 0))
        #ftlneutrino.add_solid(testobject, rotation=None, displacement=(0, 0, 0))
        ftlneutrino.add_pmt(pmt, rotation=None, displacement=(0,-0.5 - pmtpos, 0))

        ftlneutrino.flatten()
        ftlneutrino.bvh = load_bvh(ftlneutrino)
        
        # view(ftlneutrino)
        # quit()

        sim = Simulation(ftlneutrino)

        for ev in sim.simulate(photon_rays_angledy_circle(10000.0, diameter, 10.0, angle), keep_photons_beg=True, keep_photons_end=True, run_daq=True, max_steps=1000):

            detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)
            detn = np.sum(detected)
            # nohit = (ev.photons_end.flags & (0x1 << 0)).astype(bool)
            # surface_absorb = (ev.photons_end.flags & (0x1 << 3)).astype(bool)
            # bulk_absorb = (ev.photons_end.flags & (0x1 << 1)).astype(bool)
            print 'detected ' + str(detn)
            #print 'nohit ' + str(np.sum(nohit))
            #print 'surface absorb '+ str(np.sum(surface_absorb))
            #print 'bulk absorb ' + str(np.sum(bulk_absorb))
            #print 'total ' + str(np.sum(detected)+np.sum(nohit)+np.sum(surface_absorb)+np.sum(bulk_absorb))

            endpos = ev.photons_end.pos[detected]
            
            perf_rad = perf_radius(goal*detn, endpos)
            #deviation = np.sqrt(np.add((np.std(endpos[:,0]))**2, (np.std(endpos[:,2]))**2))
            deviation = find_main_deviation(perf_rad, endpos)
            print 'pmtpos: ' + str(pmtpos) + ' perf_rad: ' + str(perf_rad) + ' deviation: ' + str(deviation)

            fig = plt.figure(figsize=(7.8, 6))
            plt.scatter(ev.photons_end.pos[detected][:,0], ev.photons_end.pos[detected][:,2])
            #plt.axis((-0.7, 0.7, -0.3, 0.3))
            plt.xlabel('Final X-Position (m)')
            plt.ylabel('Final Z-Position (m)')
            plt.title('Hit Detection Locations')
            plt.show()
            
        return deviation, perf_rad
      
    def best_pmt_position(angle, num, goal, shape):
        #use this to find the comatic position
        #use shape 'pc' or 'spherical'
        perf_rad_indicator = diameter/2.0*goal
        perf_radii = -np.ones(num)
        deviations = -np.ones(num)
        if shape == 'pc':
            pmt_positions = np.linspace(0.01, pcfocal_length, num)
            #pmt_positions = np.linspace(0.8*pcfocal_length, 1.2*pcfocal_length, num, endpoint=True)
            #pmt_positions = np.linspace(2.2, 2.7, num)
            print 'pcfocal_length: ' + str(pcfocal_length)
        elif shape == 'spherical':
            #pmt_positions = np.linspace(0.01, focal_length, num)
            #pmt_positions = np.linspace(0.8*focal_length, 1.2*focal_length, num, endpoint=True)
            #pmt_positions = np.linspace(4.0, 5.5, num)
            pmt_positions = np.repeat(4.5, num)
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
            plt.text(mid, 0.43, 'goal: ' + str(goal))
            plt.text(mid, 0.41, 'perf_indicator: ' + str(perf_indicator))
            plt.text(mid, 0.39, 'best pmt pos (std): ' + str(deviation_pmt_pos))
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
            plt.text(mid, 0.39, 'best pmt pos (perfrad): ' + str(perf_radius_pmt_pos))
            plt.xlabel('PMT Position')
            plt.ylabel('Performance Radius of Final Position')
            plt.title('PC PMT Position vs Performance Radii for pcradius: ' + str(pcrad) + ' at angle: ' + str(angle))
            plt.show()

        elif shape == 'spherical':
            fig = plt.figure(figsize=(8, 8))
            plt.scatter(pmt_positions, deviations)
            plt.axis((low_bound, high_bound, 0, 0.5))
            plt.text(mid, 0.45, 'focal_length: ' + str(focal_length))
            plt.text(mid, 0.43, 'goal: ' + str(goal))
            plt.text(mid, 0.41, 'perf_indicator: ' + str(perf_indicator))
            plt.text(mid, 0.39, 'best pmt pos (std): ' + str(deviation_pmt_pos))
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
            plt.text(mid, 0.39, 'best pmt pos (perfrad): ' + str(perf_radius_pmt_pos))
            plt.xlabel('PMT Position')
            plt.ylabel('Performance Radius of Final Position')
            plt.title('PMT Position vs Performance Radii for R1: ' + str(R1) + ' R2: ' + str(R2) + ' at angle: ' + str(angle))
            plt.show()

        return deviation_pmt_pos, perf_radius_pmt_pos

    print best_pmt_position(np.radians(40.0), 7, 0.9, 'spherical')

    def flat_snells_law_test(initial_angle):
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
        
    #print 'flat', flat_snells_law_test(np.pi/5)    

    def curved_snells_law_test(xpos, pmtpos, shape): 
        #sends in a light ray at x-position xpos to the lens.  Make sure that the pmt is placed in between the two surfaces of the lens- which is at a positive y number for most lenses.  Note however that pmtpos is subtracted in ftlneutrino.add_pmt below, so make pmtpos negative to place it in the positive y direction.
        # might not work for spherical lenses-I still need to check to see if it works.

        ftlneutrino = Detector(lm.ls)
        if shape == 'pc':
            ftlneutrino.add_solid(lens_solid2)
            #ftlneutrino.add_solid(lens_solidc)
        elif shape == 'spherical':
            ftlneutrino.add_solid(lens_solidm)
            #ftlneutrino.add_solid(lens_solidm, rotation=make_rotation_matrix(-np.pi, (1,0,0)))
        else:
            print "need shape"

        ftlneutrino.add_pmt(smallpmt, rotation=None, displacement=(0,-0.5 - pmtpos, 0))

        ftlneutrino.flatten()
        ftlneutrino.bvh = load_bvh(ftlneutrino)
        
        # view(ftlneutrino)
        # quit()

        sim = Simulation(ftlneutrino)
        
        for ev in sim.simulate(single_ray([xpos, 5, 0], [0, -1, 0], 20), keep_photons_beg=True, keep_photons_end=True, run_daq=True, max_steps=1000):

            detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)
            detn = np.sum(detected)

            endpos = ev.photons_end.pos[detected]
            
            begdir = ev.photons_beg.dir[detected]
            enddir = ev.photons_end.dir[detected]
            n1 = float(lm.ls_refractive_index)
            n2 = float(lm.lensmat_refractive_index)
            if shape == 'pc':
                ypos = np.sqrt(pcrad**2-xpos**2)
            elif shape == 'spherical':
                ypos = np.sqrt(R1**2-xpos**2)
            axis = -normalize(np.array([xpos, ypos, 0]))
            initial_angle = np.arccos(np.dot(begdir[0], axis))
            predicted_final_angle = np.arcsin(n1/n2*np.sin(initial_angle))
            final_angle = np.arccos(np.dot(enddir[0], axis))
            predicted_ratio = n1/n2
            actual_ratio = np.sin(final_angle)/np.sin(initial_angle)

            print 'pcrad', pcrad
            print 'xpos', xpos
            print 'initial_angle', initial_angle
            print 'endpos: ' + str(endpos[0])
            print 'predicted final_angle', predicted_final_angle
            print 'actual final angle', final_angle
            print 'predicted_ratio n1/n2', predicted_ratio
            print 'actual_ratio sin(t2)/sin(t1)', actual_ratio
            print 'final_angle/predicted_final_angle', final_angle/predicted_final_angle
            print 'small-angle ratio: t2/t1', final_angle/initial_angle

            if final_angle == initial_angle:
                return 'miss or no refraction'
            elif (actual_ratio > 0.95*predicted_ratio) and (actual_ratio < 1.05*predicted_ratio):
                return 'good photon'
            elif (predicted_ratio**2 * 0.95 < actual_ratio) and (predicted_ratio**2 * 1.05 > actual_ratio):
                return 'twice reflected'
            else:
                return 'bad photon'

#print curved_snells_law_test(0.2, -0.01, 'pc')


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
        # plt.hist2d(ev.photons_end.pos[detected][:,0], ev.photons_end.pos[detected][:,2], bins=200)
        # plt.axis((-0.7, 0.7, -0.3, 0.3))
        # plt.xlabel('Final X-Position (m)')
        # plt.ylabel('Final Z-Position (m)')
        # plt.title('Hit Detection Locations')
        # plt.colorbar()
        # plt.show()

        # fig = plt.figure(figsize=(7.8, 6))
        # plt.scatter(ev.photons_end.pos[detected][:,0], ev.photons_end.pos[detected][:,2])
        # #plt.axis((-0.7, 0.7, -0.3, 0.3))
        # plt.xlabel('Final X-Position (m)')
        # plt.ylabel('Final Z-Position (m)')
        # plt.title('Hit Detection Locations')
        # plt.show()
