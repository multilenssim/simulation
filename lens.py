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
#import pyparsing

def lens(radius, diameter, nsteps=1024):
    #constructs a convex lens mesh
    angles = np.linspace(-np.pi/2, np.pi/2, nsteps)
    return make.rotate_extrude(diameter/2.0*np.cos(angles), np.sign(angles)*(np.sqrt(radius**2-(diameter/2.0*np.cos(angles))**2)-np.sqrt(radius**2-(diameter/2.0)**2)), nsteps=64)

def pclens(radius, diameter, nsteps=1024):
    #constructs a plano-convex lens mesh
    angles = np.linspace(-np.pi/2, np.pi/2, nsteps)
    return make.rotate_extrude(diameter/2.0*np.cos(angles), (0.5*(np.sign(angles)+1))*(np.sqrt(radius**2-(diameter/2.0*np.cos(angles))**2)-np.sqrt(radius**2-(diameter/2.0)**2)), nsteps=64)

def paralens(focal_length, diameter, nsteps=1024):
    #constructs a paraboloid lens
    height = 0.25/focal_length*(diameter/2)**2
    a = np.linspace(0, diameter/2.0, nsteps/2, endpoint=False)
    b = np.linspace(diameter/2.0, 0, nsteps/2)
    return make.rotate_extrude(np.concatenate((a, b)), np.concatenate((0.25/focal_length*(a)**2-height, -0.25/focal_length*(b)**2+height)), nsteps=64)

def pcparalens(focal_length, diameter, nsteps=1024):
    #constructs a planoconvex paraboloid lens
    height = 0.25/focal_length*(diameter/2.0)**2
    a = np.linspace(0, diameter/2.0, nsteps/2, endpoint=False)
    b = np.linspace(diameter/2.0, 0, nsteps/2)
    c = np.tile(height, nsteps/2)
    return make.rotate_extrude(np.concatenate((a, b)), np.concatenate((0.25/focal_length*(a)**2, c)), nsteps=64)

def inner_blocker_mesh(radius, thickness, nsteps=16):
    rightangles = np.linspace(np.pi, 2*np.pi/3, nsteps, endpoint=False)
    topangles = np.linspace(5*np.pi/3, 4*np.pi/3, nsteps, endpoint=False)
    leftangles = np.linspace(np.pi/3, 0, nsteps, endpoint=False)
    rightx = radius*np.cos(rightangles) + radius
    righty = radius*np.sin(rightangles) - np.sqrt(3)/3.0*radius
    topx = radius*np.cos(topangles)
    topy = radius*np.sin(topangles) + 2*radius/np.sqrt(3)
    leftx = radius*np.cos(leftangles) - radius
    lefty = radius*np.sin(leftangles) - np.sqrt(3)*radius/3.0
    xs = np.concatenate((rightx, topx, leftx))
    ys = np.concatenate((righty, topy, lefty))
    return make.linear_extrude(xs, ys, thickness)

def corner_blocker_mesh(radius, thickness, nsteps=16):
    #constructs triangular corners with a single curved side.
    angles = np.linspace(5*np.pi/6.0, np.pi/6.0, nsteps)
    bottomx = radius*np.cos(angles)
    bottomy = radius*np.sin(angles)-2*radius
    xs = np.append(bottomx, 0)
    ys = np.append(bottomy, 0)
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

def spherical_lens(R1, R2, diameter, nsteps=1024):
    '''constructs a spherical lens with specified radii of curvature. Works with meniscus lenses. If meniscus, light should go in the larger cap first for better performance- see MOE 4th ed. p.86. Make sure not to fold R1 through R2 or vica-versa in order to keep rotate_extrude going counterclockwise.
    shift is the amount needed to move the hemisphere in the y direction to make the spherical cap 
    R1 goes towards positive y, R2 towards negative y.'''
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

def pclens2(radius, diameter, nsteps=1024):
    #works best with angles endpoint=True
    halfd = diameter/2.0
    shift = np.sqrt(radius**2-(halfd)**2)
    theta = np.arctan(shift/(halfd))
    angles = np.linspace(theta, np.pi/2, nsteps)
    x1 = np.array([0, halfd])
    #xtry = np.array([0])
    y1 = np.array([0, 0])
    x2 = radius*np.cos(angles)
    y2 = radius*np.sin(angles) - shift
    xs = np.concatenate((x1, x2))
    ys = np.concatenate((y1, y2))
    return make.rotate_extrude(x2, y2, nsteps=64)

def klens(diameter, thickness, nsteps=1024):
    a = np.linspace(0, diameter/2, nsteps/2, endpoint=False)
    b = np.linspace(diameter/2, 0, nsteps/2)
    return make.rotate_extrude(np.concatenate((a, b)), np.concatenate((2*thickness/diameter**2*(a)**2-0.5*thickness, -2.0*thickness/diameter**2*(b)**2+0.5*thickness)), nsteps=64)

def cylindrical_shell(inner_radius, outer_radius, height, nsteps=1024):
    if inner_radius < outer_radius:
        return make.rotate_extrude([inner_radius, outer_radius, outer_radius, inner_radius], [-height/2.0, -height/2.0, height/2.0, height/2.0], nsteps)
    else:
        print "inner radius must be less than outer radius"

def photon_rays(diameter, n):
    #constructs photons traveling in rays parallel with the optical axis of lens
    pos = np.array([(i, 1.0, 0) for i in np.linspace(-diameter/2, diameter/2, n)])
    dir = np.tile((0, -1.0, 0), (n,1))
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000, n)
    return Photons(pos, dir, pol, wavelengths)

def photon_rays_angledy(diameter, theta, n):
    #constructs collimated photons traveling at a specified angle from the x-axis (actually I think it's from the y-axis) in the xy-plane. z_initial = 0, y_initial = 1
    #y axis is optical axis
    #theta should be in the range from -pi/2 to pi/2
    pos = np.array([(i+1.0*np.tan(theta), 1.0, 0) for i in np.linspace(-diameter/2, diameter/2, n)])
    dir = np.tile((-1.0*np.tan(theta),-1.0, 0), (n,1))
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000, n)
    return Photons(pos, dir, pol, wavelengths)

def photon_rays_angledx_circle(diameter, theta, n):
    #constructs collimated photons traveling in a uniform circle at a specified angle from the x-axis (maybe actually y-axis) in the xy-plane. z_initial = 0, y_initial = 1
    #theta should be in the range from -pi/2 to pi/2
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

def photon_rays_angledz(diameter, theta, n):
    #constructs collimated photons traveling at a specified angle from the y axis in the yz plane. y-initial=1
    #theta ranges from -pi/2 to pi/2
    pos = np.array([(i, 1.0, np.tan(theta)) for i in np.linspace(-diameter/2, diameter/2, n)])
    dir = np.tile((0,-1.0,-np.tan(theta)), (n,1))
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000, n)
    return Photons(pos, dir, pol, wavelengths)

def photon_circle(n1, n2, diameter, dir=(0,-1,0)):
    '''constructs a circle of photons with higher density at the center
    this circle has repeat values at (0, 1, 0)
    for n=10,000: n1= 100, n2=100'''
    n = n1*n2
    pos = np.array([(r*np.cos(theta), 1.0, r*np.sin(theta)) for theta in np.linspace(0.0, 2*np.pi, n1, endpoint=False) for r in np.linspace(0.0, diameter/2, n2)])
    dir = np.tile(dir, (n,1))
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000, n)
    return Photons(pos, dir, pol, wavelengths)

def photon_circle2(n1, n2, diameter, dir=(0,-1,0)):
    '''constructs a circle of photons with higher density at the center
    this circle DOES NOT have repeat values at (0, 1, 0)
    for n=10,000: n1=101, n2=100. unique_rows from: stackoverflow.com/questions/8560440, user545424'''
    n = n1*n2-n1+1
    def unique_rows(a):
        a = np.ascontiguousarray(a)
        unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
        return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
    pos = unique_rows(np.array([(r*np.cos(theta), 1.0, r*np.sin(theta)) for theta in np.linspace(0.0, 2*np.pi, n1, endpoint=False) for r in np.linspace(0.0, diameter/2, n2)]))
    dir = np.tile(dir, (n,1))
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000, n)
    return Photons(pos, dir, pol, wavelengths)

def uniform_circle(n, diameter, dir=(0,-1.0,0)):
    #constructs a uniform disk of photons.  See disk point picking @ wolfram.com.
    radii = np.random.uniform(0, diameter/2.0, n)
    angles = np.random.uniform(0, 2*np.pi, n)
    points = np.empty((n,3))
    points[:,0] = np.sqrt(diameter/2.0)*np.sqrt(radii)*np.cos(angles)
    points[:,1] = np.repeat(1.0, n)
    points[:,2] = np.sqrt(diameter/2.0)*np.sqrt(radii)*np.sin(angles)
    pos = points
    dir = np.tile(dir, (n,1))
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000, n)
    return Photons(pos, dir, pol, wavelengths)

def photon_gauss(pos, sigma, n):
    #constructs an initial distribution of photons with uniform angular position and radius determined by a normal distribution. Each photon is launched in a random direction.
    radii = np.random.normal(0.0, sigma, n)
    angles = np.linspace(0.0, np.pi, n, endpoint=False)
    points = np.empty((n,3))
    points[:,0] = radii*np.cos(angles) + pos[0]
    points[:,1] = np.tile(pos[1], n)
    points[:,2] = radii*np.sin(angles) + pos[2]
    pos = points
    dir = uniform_sphere(n)
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(1000.0, n)
    return Photons(pos, dir, pol, wavelengths) 

def focal_length_finder(diameter, thickness):
    x = 0.25
    n = 1.5
    N = 2.0
    a = 2*thickness/diameter**2
    H = a/4*diameter**2
    u = np.arctan(2*a*x)
    m = -np.tan(np.pi/2-u+np.arcsin(n/N*np.sin(u)))
    b = a*x**2-m*x-H
    X = (m+np.sqrt(m**2+4*a*(-a*x**2+m*x+2*H)))/(-2*a)
    p = np.arctan((2*a*m*X-1)/(2*a*X+m))
    q = np.arcsin(N/n*np.sin(p))
    M = (1+2*a*X*np.tan(q))/(2*a*X-np.tan(q))
    focal_length = -a*X**2+H-M*X

    if b <= H:
        print "bad focal length"
    return focal_length

if __name__ == '__main__':
    from chroma.sim import Simulation
    from chroma import sample
    from chroma.event import Photons
    from chroma.loader import load_bvh
    from chroma.generator import vertex
    from chroma.io.root import RootWriter
    import matplotlib.pyplot as plt

    #defining the specifications of the meshes
    lens1 = lens(5.0, 1.0)
    lens2 = pclens2(1.0, 1.0)
    lens3 = paralens(0.5, 1.0)
    lens4 = pcparalens(0.5,1.0)
    lens5 = make.box(1.0,1.0,1.0)
    THICK = 0.1
    lensk = klens(1.0, THICK)
    lensm = spherical_lens(-2, -1, 1.0)
    
    #creating solids of form (mesh, inside_mat, outside_mat)
    lens_solid1 = Solid(lens1, lm.lensmat, lm.ls)
    lens_solid2 = Solid(lens2, lm.lensmat, lm.ls)
    lens_solid3 = Solid(lens3, lm.lensmat, lm.ls)
    lens_solid4 = Solid(lens4, lm.lensmat, lm.ls)
    lens_solid5 = Solid(lens5, lm.lensmat, lm.ls)
    lens_solidk = Solid(lensk, lm.lensmat, lm.ls)
    lens_solidm = Solid(lensm, lm.lensmat, lm.ls)
    
    pmt = Solid(make.box(10,1,10), glass, lm.ls, surface=lm.fulldetect, color=0x00ff00)
    testsolid = Solid(make.box(0.05, 0.05, 0.05), glass, lm.ls, surface=lm.fulldetect, color=0x00ff00)
    blocker = Solid(cylindrical_shell(0.5, 10.0, 0.000001), glass, lm.ls, surface=black_surface, color=0xff0000)

#testing making a square filled with glass then a circle of vacuum inside that
    filled_blocker = Solid(make.box(20.0,0.000001,20.0), lm.blackhole, lm.ls)
    
    anglez = np.linspace(0.0,2*np.pi,4096.0,endpoint=False)
    
    unfilled_circle=Solid(make.rotate_extrude([0,10.0,10.0,0],[0.01,0.01,-0.01,-0.01]), vacuum, lm.ls)

    print THICK
    print focal_length_finder(1.0, THICK)
    #creates the entire Detector and adds the solid & pmt
    ftlneutrino = Detector(lm.ls)
    #ftlneutrino.add_solid(blocker)
    outer_blocker = Solid(outer_blocker_mesh(1.5, 0.2), lm.lensmat, lm.ls)
    ftlneutrino.add_solid(outer_blocker)
    ftlneutrino.add_solid(testsolid, displacement=(0, 0, 0))
    #ftlneutrino.add_solid(lens_solid2)
    #ftlneutrino.add_pmt(pmt, rotation=None, displacement=(0,-0.5 -5, 0))
    #-2.02276 is the calculated value for paralens(0.5, 1.0)
    #-8.14654 is y value for lens1 with focal length 7.65654, box of sides: 1.0, R=5, d=1, T=0.0501256
    ftlneutrino.flatten()
    ftlneutrino.bvh = load_bvh(ftlneutrino)

    #focal_length_finder(1.0, THICK)*1.5
    view(ftlneutrino)
    quit() 

    import sys
    sys.exit("End here")
    
    #f = RootWriter('test.root')

    sim = Simulation(ftlneutrino)
   
    ftlphotons = photon_rays(1.0, 10000.0)
    ftlphotonsx = photon_rays_angledy(1.0, np.pi/4, 10000.0)
    ftlphotonsxcir = photon_rays_angledy_circle(1.0, np.pi/3, 10000.0)
    ftlphotonsz = photon_rays_angledz(1.0, np.pi/4, 10000.0)
    cirphotons = photon_circle(100.0, 100.0, 1.0)
    cirphotons2 = photon_circle2(101.0, 100.0, 1.0)
    uniformcir = uniform_circle(10000.0, 1.0)
    photongauss = photon_gauss((0.0, 15.0, 1.5), 0.01, 1000000.0)

    for ev in sim.simulate(photongauss, keep_photons_beg=True, keep_photons_end=True, run_daq=True, max_steps=100):

        detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)
      
    #     #f.write_event(ev) 

    #     #prints standard deviation of ending x position, z position, and radius
    #     print np.std(ev.photons_end.pos[detected][:,0]), np.std(ev.photons_end.pos[detected][:,2]), np.std(np.sqrt((ev.photons_end.pos[detected][:,0])**2 + (ev.photons_end.pos[detected][:,2])**2))
        
        # fig = plt.figure(figsize=(6, 6))
        # plt.hist(ev.photons_end.pos[detected][:,0],100)
        # plt.xlabel('X Position (m)')
        # plt.ylabel('Number of Photons')
        # plt.title('Hit X-Locations')
        
        # fig = plt.figure(figsize=(6, 6))
        # plt.hist(ev.photons_end.pos[detected][:,2],100)
        # plt.xlabel('Z Position (m)')
        # plt.ylabel('Number of Photons')
        # plt.title('Hit Z-Locations')
        
        # fig = plt.figure(figsize=(6, 6))
        # plt.scatter(ev.photons_beg.pos[detected][:,0], ev.photons_end.pos[detected][:,0])
        # plt.xlabel('Initial X-Position (m)')
        # plt.ylabel('Finial X-Position (m)')
        # plt.title('Initial vs Final X-Location')
        
        # fig = plt.figure(figsize=(6, 6))
        # plt.scatter(ev.photons_beg.pos[detected][:,2], ev.photons_end.pos[detected][:,2])
        # plt.xlabel('Initial Z-Position (m)')
        # plt.ylabel('Finial Z-Position (m)')
        # plt.title('Initial vs Final Z-Location')
        
        # fig = plt.figure(figsize=(6, 6))
        # plt.scatter(ev.photons_beg.pos[detected][:,0], ev.photons_beg.pos[detected][:,2])
        # plt.xlabel('Initial X-Position (m)')
        # plt.ylabel('Initial Z-Position (m)')
        # plt.title('Initial Location Plot')
        
        # fig = plt.figure(figsize=(6, 6))
        # plt.scatter(ev.photons_end.pos[detected][:,0], ev.photons_end.pos[detected][:,2])
        # plt.xlabel('Final X-position (m)')
        # plt.ylabel('Final Z-position (m)')
        # plt.title('Final Location Plot')
        
        # fig = plt.figure(figsize=(6, 6))
        # plt.scatter(np.sqrt((ev.photons_beg.pos[detected][:,0])**2+(ev.photons_beg.pos[detected][:,2])**2), np.sqrt((ev.photons_end.pos[detected][:,0])**2+(ev.photons_end.pos[detected][:,2])**2))
        # plt.xlabel('Initial Radius (m)')
        # plt.ylabel('Final Radius (m)')
        # plt.title('Initial vs. Final Radius Plot')

        # fig = plt.figure(figsize=(7.8, 6))
        # plt.hist2d(ev.photons_beg.pos[detected][:,0], ev.photons_beg.pos[detected][:,2], bins=100)
        # plt.xlabel('Initial X-Position (m)')
        # plt.ylabel('Initial Z-Position (m)')
        # plt.title('Initial Position')
        # plt.colorbar()
        
        fig = plt.figure(figsize=(7.8, 6))
        plt.hist2d(ev.photons_end.pos[detected][:,0], ev.photons_end.pos[detected][:,2], bins=100, range=[[-0.15, 0.15], [-1.5, -1.1]])
        plt.xlabel('Final X-Position (m)')
        plt.ylabel('Final Z-Position (m)')
        plt.title('Hit Detection Locations')
        plt.colorbar()

        fig = plt.figure(figsize=(7.8, 6))
        plt.hist2d(ev.photons_end.pos[detected][:,0], ev.photons_end.pos[detected][:,2], bins=100)
        plt.xlabel('Final X-Position (m)')
        plt.ylabel('Final Z-Position (m)')
        plt.title('Hit Detection Locations')
        plt.colorbar()

        plt.show()
        
    #f.close()

