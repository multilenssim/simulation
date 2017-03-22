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
from scipy import optimize, ndimage, spatial
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

def pclens2(radius, diameter, nsteps=128):
    #works best with angles endpoint=True
    halfd = diameter/2.0
    shift = np.sqrt(radius**2-(halfd)**2)
    theta = np.arctan(shift/(halfd))
    angles = np.linspace(theta, np.pi/2, nsteps)
    x2 = radius*np.cos(angles)
    y2 = radius*np.sin(angles) - shift
    xs = np.concatenate((np.zeros(1), x2))
    ys = np.concatenate((np.zeros(1), y2))
    return make.rotate_extrude(xs, ys, nsteps=64)
    
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

def spherical_lens(R1, R2, diameter, nsteps=64):
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
    return make.rotate_extrude(np.concatenate((x2,x1)), np.concatenate((y2,y1)), nsteps=64)
    
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
    pcfocal_length = 1.0/((n2/n1-1.0)*(1.0/radius))
    return pcfocal_length

def spherical_focal_length(R1, R2, diameter):
    #lensmaker's equation
    shift1 = -np.sign(R1)*np.sqrt(R1**2 - (diameter/2.0)**2)
    shift2 = -np.sign(R2)*np.sqrt(R2**2 - (diameter/2.0)**2)
    thickness = abs((R1+shift1)-(R2+shift2))
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

    R1 = 1.426
    R2 = 2.384
    #shiftin = 9.72368
    shiftin = 10.0
    focal_length = spherical_focal_length(R1, R2, diameter)
    print focal_length

    #focal_length = 6.0
    #R2 = findR2(R1, diameter, focal_length)
    #actual_f_l = spherical_focal_length(R1, R2, diameter)
    
    pcrad = 0.9
    #shiftin = 0
    #pcfocal_length = pc_focal_length(pcrad)
    #print pcfocal_length

    #defining the specifications of the meshes
    lens1 = lens(5.0, diameter)
    lens2 = pclens2(pcrad, diameter)
    lens3 = paralens(0.5, diameter)
    lens4 = pcparalens(0.5, diameter)
    lens5 = make.box(1.0,1.0,1.0)
    lensm = spherical_lens(R1, R2, diameter)
    
    #creating solids of form (mesh, inside_mat, outside_mat)
    lens_solid1 = Solid(lens1, lm.lensmat, lm.ls)
    lens_solid2 = Solid(lens2, lm.lensmat, lm.ls)
    #lens_solid2 = Solid(lens2, lm.lensmat, lm.ls, lm.noreflect)
    lens_solid3 = Solid(lens3, lm.lensmat, lm.ls)
    lens_solid4 = Solid(lens4, lm.lensmat, lm.ls)
    lens_solid5 = Solid(lens5, lm.lensmat, lm.ls)
    lens_solidm = Solid(lensm, lm.lensmat, lm.ls)


    pmt = Solid(make.box(100.0,1.0,100.0), glass, lm.ls, surface=lm.fulldetect, color=0x00ff00)
    blocker = Solid(cylindrical_shell(diameter/2.0, 100.0, 0.00001), glass, lm.ls, black_surface, 0xff0000)
    #testobject = Solid(make.box(0.05, 0.05, 0.05), glass, lm.ls, color=0xff0000)
    #ultraabsorb = Solid(make.box(100, 100, 100), lm.ls, lm.ls, surface = lm.fulldetect)

    #creates the entire Detector and adds the solid & pmt
    ftlneutrino = Detector(lm.ls)
    #ftlneutrino.add_solid(ultraabsorb)
    #ftlneutrino.add_solid(lens_solid2)
    ftlneutrino.add_solid(lens_solidm)
    ftlneutrino.add_solid(blocker)
    #ftlneutrino.add_solid(testobject, rotation=None, displacement=(0, -1, 0))
    #ftlneutrino.add_pmt(pmt, rotation=None, displacement=(0,-0.5-3.0, 0))
    ftlneutrino.add_pmt(pmt, rotation=None, displacement=(0,-0.5-focal_length + shiftin, 0))
    #ftlneutrino.add_pmt(pmt, rotation=None, displacement=(0,-0.5-pcfocal_length + shiftin, 0))
    #-2.02276 is the calculated value for paralens(0.5, 1.0)
    #-8.14654 is y value for lens1 with focal length 7.65654, box of sides: 1.0, R=5, d=1, T=0.0501256
    ftlneutrino.flatten()
    ftlneutrino.bvh = load_bvh(ftlneutrino)
    
    # from chroma.io.root import RootWriter
    # f = RootWriter('test.root')

    # view(ftlneutrino)
    # quit()

    #performance_angle represents the last angle who's deviation is less than 0.5. Thus increasing  the performance angle increases the performance of the lens at focusing light from higher incident angles. goal*detn = target amount of photons for the perf_radius function.  (detn is the amount of detected photons.)
    num = 10
    angles = np.linspace(0, 1.4, num)
    deviations = np.zeros(num)
    perf_radii = np.zeros(num)
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
    for i in range(num):
        #angle = np.radians(45)
        angle = angles[i]
        for ev in sim.simulate(photon_rays_angledy_circle(10000.0, diameter, 5.0, angle), keep_photons_beg=True, keep_photons_end=True, run_daq=True, max_steps=1000):

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
            radii = np.sqrt(np.add(endpos[:,0]**2, endpos[:,2]**2))
            radstd = np.std(radii)
            deviations[i] = deviation
            if deviation <= perf_indicator:
               performance_angle = angle
               performance_index = i

            # fig = plt.figure(figsize=(7.8, 6))
            # plt.hist(radii, bins=np.linspace(-2, 2, 150))
            # plt.axis((-2.0, 2.0, 0, 3000))
            # plt.text(-1.5, 2500, 'standard deviation of radius: ' + str(deviation))
            # plt.xlabel('Final Radius (m)')
            # plt.ylabel('Amount of Photons')
            # plt.title('Hit Detection Locations')
            # plt.show()

            xstd = np.std(endpos[:,0])
             
            perf_radii[i] = perf_rad
            if perf_rad <= perf_rad_indicator:
                perf_rad_angle = angle
            
            #f.write_event(ev)
    
    print 'Done'

    radpre45_sum = 0
    radpost45_sum = 0
    for i in range(num):
        if angles[i] < np.pi/4:
            radpre45_sum += perf_radii[i]
        else:
            radpost45_sum += perf_radii[i]
    radtotal_sum = radpre45_sum + radpost45_sum

    pre45_sum = 0
    post45_sum = 0
    for i in range(num):
        if angles[i] < np.pi/4:
            pre45_sum += deviations[i]
        else:
            post45_sum += deviations[i]
    total_sum = pre45_sum + post45_sum

    # print 'performance_index: ' + str(performance_index)
    
    if shiftin == 0:
        shift_string = ''
    elif shiftin > 0:
        shift_string = ' with pmt shift-in: ' + str(shiftin)
    else:
        shift_string = ' with pmt shift-out: ' + str(shiftin)
    
    #plot for spherical lenses:
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(angles, deviations)
    plt.axis((0, 1.1, 0, 3))
    plt.text(0.25, 2.9, 'focal length: ' + str(focal_length))
    plt.text(0.25, 2.8, 'goal: ' + str(goal))
    plt.text(0.25, 2.7, 'performance indicator: ' + str(perf_indicator))
    plt.text(0.25, 2.6, 'performance angle: ' + str(performance_angle))
    plt.text(0.25, 2.5, 'pre45_sum: ' + str(pre45_sum))
    plt.text(0.25, 2.4, 'post45_sum: ' + str(post45_sum))
    plt.text(0.25, 2.3, 'total_sum: ' + str(total_sum))
    plt.xlabel('Incident Angle')
    plt.ylabel('Deviation')
    plt.title('Incident Angle vs Deviation for R1: ' + str(R1) + ' R2 ' + str(R2) + ' and focal_length: ' + str(focal_length) + shift_string)
    plt.show()
        
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(angles, perf_radii)
    plt.axis((0, 1.1, 0, 3))
    plt.text(0.25, 2.9, 'focal length: ' + str(focal_length))
    plt.text(0.25, 2.8, 'goal: ' + str(goal))
    plt.text(0.25, 2.7, 'perf_rad_indicator: ' + str(perf_rad_indicator))
    plt.text(0.25, 2.6, 'perf_rad_angle: ' + str(perf_rad_angle))
    plt.text(0.25, 2.5, 'radpre45_sum: ' + str(radpre45_sum))
    plt.text(0.25, 2.4, 'radpost45_sum: ' + str(radpost45_sum))
    plt.text(0.25, 2.3, 'radtotal_sum: ' + str(radtotal_sum))
    plt.xlabel('Incident Angle')
    plt.ylabel('Performance Radius')
    plt.title('Incident Angle vs perf_radius for R1: ' + str(R1) + ' R2 ' + str(R2) + ' and focal_length: ' + str(focal_length) + shift_string)
    plt.show()

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

    fig = plt.figure(figsize=(7.8, 6))
    plt.hist(ev.photons_end.pos[detected][:,0], bins=np.linspace(-2, 2, 150))
    plt.axis((-2.0, 2.0, 0, 3000))
    plt.text(-1.5, 2500, 'standard deviation of xpos: ' + str(xstd))
    plt.xlabel('Final X-Position (m)')
    plt.ylabel('Amount of Photons')
    plt.title('Hit Detection Locations')
    plt.show()

    # fig = plt.figure(figsize=(7.8, 6))
    # plt.hist2d(ev.photons_end.pos[detected][:,0], ev.photons_end.pos[detected][:,2], bins=100)
    # #plt.axis((-2.0, 2.0, -0.5, 0.5))
    # plt.xlabel('Final X-Position (m)')
    # plt.ylabel('Final Z-Position (m)')
    # plt.title('Hit Detection Locations')
    # plt.colorbar()
    # plt.show()

    # fig = plt.figure(figsize=(7.8, 6))
    # plt.scatter(ev.photons_end.pos[detected][:,0], ev.photons_end.pos[detected][:,2])
    # plt.axis((-2.0, 2.0, -0.5, 0.5))
    # plt.xlabel('Final X-Position (m)')
    # plt.ylabel('Final Z-Position (m)')
    # plt.title('Hit Detection Locations')
    # plt.show()
    
    #f.close()
