import numpy as np


"""Spherical:"""

"""
f = .5
D = .5 ???
n_ls = 1.5
n_lens = 2
n = n_lens / n_ls = 4/3
thickness:  T = 2*radius-2*np.sqrt(radius**2-(diameter/2)**2) 
T=1/100  ==>  R=33/100 ==> diameter=np.sqrt(131)/100 ==> f=0.5
R=5 , d=1.0 ==> T=0.0501256 ==> f=7.64654
R = f*(n-1)*(1+np.sqrt(1-T/(f*n)))
"""

"""Parabolic:"""

f = 1.25
d = 1.0
x = 0.25
n = 1.5
N = 2.0

#thickness = (1/(4*f))/2*d**2

a=1/(4*f)

#b must be > H !!!
#u=theta1, p=theta3, q=theta4 

H = a/4*d**2
u = np.arctan(2*a*x)
m = -np.tan(np.pi/2-u+np.arcsin(n/N*np.sin(u)))
b = a*x**2-m*x-H
X = (m+np.sqrt(m**2+4*a*(-a*x**2+m*x+2*H)))/(-2*a)
p = np.arctan((2*a*m*X-1)/(2*a*X+m))
q = np.arcsin(N/n*np.sin(p))
M = (1+2*a*X*np.tan(q))/(2*a*X-np.tan(q))
F = -a*X**2+H-M*X

if b > H:
    print F
else:
    print "b should be greater than H, b=",b,"H=",H

#B is very close to 3f with d=1!



'''
def find_vertex(a,b,c,d,e):
    pent_center = (a+b+c+d+e)/5
    height_from_center = np.sqrt(E**2/3-np.dot(pent_center-a, pent_center-a))
    vertex = height_from_center*normalize(pent_center)+pent_center 
    print vertex
    print E/2*phi, E/2

find_vertex(facecoords[8], facecoords[10], facecoords[6], facecoords[18], facecoords[4])
'''


'''#code for blocker_mesh:  NOT USED
def blocker_mesh(diameter, outer_width, thickness, nsteps=66):
    """builds a triangular blocker with a hole of specified diameter, outer width, and thickness.  Outer width is shortest distance from hole to perimeter of triangle
    s = side length of triangle"""
    angles = np.linspace(np.pi/6, 13*np.pi/6, nsteps, endpoint=False)
    s = np.sqrt(3)*(diameter+2*outer_width)
    a = s/(2*np.sqrt(3)*np.cos(np.linspace(np.pi/6, np.pi/2, nsteps/6, endpoint=False)-np.pi/6))
    b = s/(2*np.sqrt(3)*np.cos(5*np.pi/6-np.linspace(np.pi/2, 5*np.pi/6, nsteps/6, endpoint=False)))
    c = s/(2*np.sqrt(3)*np.cos(np.linspace(5*np.pi/6, 7*np.pi/6, nsteps/6, endpoint=False)-5*np.pi/6))
    d = s/(2*np.sqrt(3)*np.cos(9*np.pi/6-np.linspace(7*np.pi/6, 9*np.pi/6, nsteps/6, endpoint=False)))
    e = s/(2*np.sqrt(3)*np.cos(np.linspace(9*np.pi/6, 11*np.pi/6, nsteps/6, endpoint=False)-9*np.pi/6))
    f = s/(2*np.sqrt(3)*np.cos(13*np.pi/6-np.linspace(11*np.pi/6, 13*np.pi/6, nsteps/6, endpoint=False)))
    triangle_radii = np.concatenate((a, b, c, d, e, f))
    return make.linear_extrude(triangle_radii*np.cos(angles), triangle_radii*np.sin(angles), thickness)'''
