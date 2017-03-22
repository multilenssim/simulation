from chroma import make, view
from chroma.geometry import Geometry, Material, Mesh, Solid, Surface
from chroma.demo.optics import glass, black_surface
from chroma.detector import Detector
from chroma.pmt import build_pmt
from chroma.event import Photons
from chroma.sim import Simulation
from chroma.sample import uniform_sphere
from chroma.transform import make_rotation_matrix, normalize
import lensmaterials as lm
import numpy as np

def lens(diameter, thickness, nsteps=8):
    a = np.linspace(0, diameter/2, nsteps/2, endpoint=False)
    b = np.linspace(diameter/2, 0, nsteps/2)
    return make.rotate_extrude(np.concatenate((a, b)), np.concatenate((2*thickness/diameter**2*(a)**2-0.5*thickness, -2*thickness/diameter**2*(b)**2+0.5*thickness)), nsteps=8)

def cylindrical_shell(inner_radius, outer_radius, thickness, nsteps=8):
    #make sure that nsteps is the same as that of rotate extrude in lens
    if inner_radius < outer_radius:
        return make.rotate_extrude([inner_radius, outer_radius, outer_radius, inner_radius], [-thickness/2.0, -thickness/2.0, thickness/2.0, thickness/2.0], nsteps)
    else:
        print "inner radius must be less than outer radius"

def inner_blocker(radius, thickness, n):
    #input radius of circles to create blocker shape for a third of the space inside of three tangent circles
    h = np.sqrt(3)/12*radius
    a = np.linspace(-radius/2.0, radius/2.0, 4, endpoint=False)
    b = np.linspace(radius/2.0, -radius/2.0, n, endpoint=False)
    y = np.concatenate((np.linspace(h, -h, 2, endpoint=False), np.linspace(-h, h, 2, endpoint=False), -np.sqrt(radius**2-b**2)+7*np.sqrt(3)/12*radius))
    return make.linear_extrude(np.concatenate((a,b)), y, thickness)

def outer_blocker(radius, thickness):
    #constructs trapezoidal blockers for the perimeter of the side
    return make.linear_extrude([-radius/np.sqrt(3), -2/np.sqrt(3)*radius, 2/np.sqrt(3)*radius, radius/np.sqrt(3)], [0, -radius, -radius, 0], thickness)

def corner_blocker(radius, thickness):
    #constructs triangle corners for a side
    return make.linear_extrude([0, -radius/np.sqrt(3), radius/np.sqrt(3)], [2.0/3*radius, -radius/3.0, -radius/3.0], thickness)
    

def build_side(E, n, d, t, blockers=True, b=1.0/1000):
    """input edge length of icosahedron 'E', the number of small triangles in the base of each face 'n', the ratio of the diameter of each lens to the maximum diameter possible 'd', the ratio of the thickness of the lens to the chosen diameter 't', and the ratio of the thickness of the blockers to that of the lenses 'b' to return a face of kabamland. First, a list of index coordinates are created for each hexagon within a tiled triangle. These are then transformed into the actual coordinate positions for the lenses based on the parameters.
    """
    E = float(E)
    R = E/(2*np.sqrt(3)*n)
    phi = (1+np.sqrt(5))/2

    key = np.empty(3*n-2)
    for i in np.linspace(0, 3*(n-1), 3*n-2):
        key[i] = n-i+2*np.floor(i/3)
    xindices = np.linspace(0, 2*(n-1), n)
    yindices = np.repeat(0,n)
    for i in np.linspace(1, 3*(n-1), 3*(n-1)):
        xindices = np.concatenate((xindices, np.linspace(n-key[i], n+key[i]-2, key[i])))
        yindices = np.concatenate((yindices, np.repeat(i, key[i])))
    xcoords = E/(2*n)*(xindices+1)-E/2
    ycoords = R*(yindices+1)-E/(2*np.sqrt(3))
    
    
    facecoords = np.array([[E/2, E/2, E/2], [E/2, E/2, -E/2], [E/2, -E/2, E/2], [E/2, -E/2, -E/2], [-E/2, E/2, E/2], [-E/2, E/2, -E/2], [-E/2, -E/2, E/2], [-E/2, -E/2, -E/2], [0.0, E/(2*phi), E*phi/2], [0.0, E/(2*phi), -E*phi/2], [0.0, -E/(2*phi), E*phi/2], [0.0, -E/(2*phi), -E*phi/2], [E/(2*phi), E*phi/2, 0.0], [E/(2*phi), -E*phi/2, 0.0], [-E/(2*phi), E*phi/2, 0.0], [-E/(2*phi), -E*phi/2, 0.0], [E*phi/2, 0.0, E/(2*phi)], [E*phi/2, 0.0, -E/(2*phi)], [-E*phi/2, 0.0, E/(2*phi)], [-E*phi/2, 0.0, -E/(2*phi)]])

    direction = -normalize(facecoords)
    axis = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    angle = np.arccos(direction[:,2])

    def tiltchange(xshift, yshift, k):
        tiltmag = (-xshift*facecoords[k,0]-yshift*facecoords[k,1])/xyfacedist
        xtilt = tiltmag*(1-np.cos(tilt))*facecoords[k,0]/xyfacedist
        ytilt = tiltmag*(1-np.cos(tilt))*facecoords[k,1]/xyfacedist
        ztilt = tiltmag*np.sin(tilt)
        return tiltmag, xtilt, ytilt, ztilt
    kabamland = Detector(lm.ls)
    
    
    for k in np.linspace(0, 3, 4):
        about_face = make_rotation_matrix(angle[k], axis[k])
        xyfacedist = np.sqrt(facecoords[k,0]**2+facecoords[k,1]**2)
        tilt = np.arccos(facecoords[k,2]/np.linalg.norm(facecoords[k]))
    
        for i in np.linspace(0, 3*n*(n-1)/2, (3*n**2-3*n+2)/2):
            tiltmag, xtilt, ytilt, ztilt = tiltchange(xcoords[i], ycoords[i], k)
            kabamland.add_solid(Solid(lens(2*d*R, 2*d*R*t), lm.lensmat, lm.ls), rotation=np.dot(about_face, make_rotation_matrix(np.pi/2, (1, 0, 0))), displacement=(xcoords[i] + facecoords[k,0] + xtilt, ycoords[i] + facecoords[k,1] + ytilt, facecoords[k,2] + ztilt))
        
        if blockers:
            blocker_thickness = 2*R*d*t*b

            '''for x in np.linspace(1, 2*n-3, n-1):
                tiltmag, xtilt, ytilt, ztilt = tiltchange((E/(2.0*n)*(x+1) - E/2.0), (R-E/(2*np.sqrt(3))), k)
                kabamland.add_solid(Solid(outer_blocker(R, blocker_thickness), lm.lensmat, lm.ls, black_surface, 0xff0000), rotation=about_face, displacement=(E/(2.0*n)*(x+1) - E/2.0 + facecoords[k,0] + xtilt, R - E/(2*np.sqrt(3)) + facecoords[k,1] + ytilt, facecoords[k,2] + ztilt))

                tiltmag, xtilt, ytilt, ztilt = tiltchange(E/(2.0*n)*(0.5*x+1) - E/2.0, R*(3/2.0*x+1) - E/(2*np.sqrt(3)), k)
                kabamland.add_solid(Solid(outer_blocker(R, blocker_thickness), lm.lensmat, lm.ls, black_surface, 0xff0000), rotation=np.dot(about_face, make_rotation_matrix(2*np.pi/3, (0, 0, 1))), displacement=(E/(2.0*n)*(0.5*x+1)-E/2.0+facecoords[k,0] + xtilt, R*(3/2.0*x+1) - E/(2*np.sqrt(3)) + facecoords[k,1] + ytilt, facecoords[k,2] + ztilt))

                tiltmag, xtilt, ytilt, ztilt = tiltchange(E/(2.0*n)*(-0.5*x+2*n-1)-E/2.0, R*(3/2.0*x+1)-E/(2*np.sqrt(3)), k)
                kabamland.add_solid(Solid(outer_blocker(R, blocker_thickness), lm.lensmat, lm.ls, black_surface, 0xff0000), rotation=np.dot(about_face, make_rotation_matrix(4*np.pi/3, (0, 0, 1))), displacement=(E/(2.0*n)*(-0.5*x+2*n-1) - E/2.0 + facecoords[k,0] + xtilt, R*(3/2.0*x+1) - E/(2*np.sqrt(3)) + facecoords[k,1] + ytilt, facecoords[k,2] + ztilt))'''

            for theta in np.linspace(np.pi/2, 11*np.pi/6, 3):
                tiltmag, xtilt, ytilt, ztilt = tiltchange(2*R*(n-1/3.0)*np.cos(theta), 2*R*(n-1/3.0)*np.sin(theta), k)
                kabamland.add_solid(Solid(corner_blocker(R, blocker_thickness), lm.lensmat, lm.ls, black_surface, 0xff0000), rotation=about_face, displacement=(2*R*(n-1/3.0)*np.cos(theta) + facecoords[k,0] + xtilt, 2*R*(n-1/3.0)*np.sin(theta) + facecoords[k,1] + ytilt, facecoords[k,2] +ztilt))

            for i in np.linspace(0, 3*n*(n-1)/2, (3*n**2-3*n+2)/2):
                '''for j in np.linspace(1, 11, 6): 
                    tiltmag, xtilt, ytilt, ztilt = tiltchange(xcoords[i] + 7*np.sqrt(3)/12*R*np.cos(3*np.pi/2-j*np.pi/6), ycoords[i] + 7*np.sqrt(3)/12*R*np.sin(3*np.pi/2-j*np.pi/6), k)
                    kabamland.add_solid(Solid(inner_blocker(R, blocker_thickness, 8), lm.lensmat, lm.ls, black_surface, 0xff0000), rotation=np.dot(about_face, make_rotation_matrix(j*np.pi/6, (0, 0, 1))), displacement=(xcoords[i] + 7*np.sqrt(3)/12*R*np.cos(3*np.pi/2-j*np.pi/6) + facecoords[k,0] + xtilt, ycoords[i] + 7*np.sqrt(3)/12*R*np.sin(3*np.pi/2-j*np.pi/6) + facecoords[k,1] + ytilt, facecoords[k,2] + ztilt))
                    '''
            if d < 1:
                tiltmag, xtilt, ytilt, ztilt = tiltchange(xcoords[i], ycoords[i], k)
                kabamland.add_solid(Solid(cylindrical_shell(d*R, R, blocker_thickness), lm.lensmat, lm.ls, black_surface, 0xff0000), rotation=np.dot(about_face, make_rotation_matrix(np.pi/2, (1,0,0))), displacement=(xcoords[i] + facecoords[k,0] + xtilt, ycoords[i] + facecoords[k,1] + ytilt, facecoords[k,2] + ztilt))
    
    view(kabamland)
    
if __name__ == '__main__':
    from chroma import sample
    from chroma.generator import vertex
    import matplotlib.pyplot as plt

    build_side(10, 3, 1.0, 0.1, blockers=True)
