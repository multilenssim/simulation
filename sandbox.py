import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.rcsetup as rcset 
from chroma.transform import make_rotation_matrix, normalize
from chroma import view, make
from chroma.stl import mesh_from_stl
import lensmaterials as lm
from scipy import integrate, optimize, ndimage, spatial

'''
         blocker_face = Solid(mh.shift(corner_blocker(R, blocker_thickness), (0, 2*R*(n-1/3.0), 0)), lm.lensmat, lm.ls, black_surface, 0xff0000)

            #xshift[i] = xcoords[i]
            #yshift[i] = ycoords[i] 
            

            xyfacedist = np.sqrt(facecoords[k,0]**2+facecoords[k,1]**2)
            facedist = np.linalg.norm(facecoords[k])
            tiltmag = (-xshift*facecoords[k,0]-yshift*facecoords[k,1])/xyfacedist
            tilt = np.arccos(facecoords[k,2]/facedist)
            xtilt = tiltmag*(1-np.cos(tilt))*facecoords[k,0]/xyfacedist
            ytilt = tiltmag*(1-np.cos(tilt))*facecoords[k,1]/xyfacedist
            ztilt = tiltmag*np.sin(tilt)'''

'''icox = np.array([E/2, E/2, E/2, E/2, -E/2, -E/2, -E/2, -E/2, 0.0, 0.0, 0.0, 0.0, E/(2*phi), E/(2*phi), -E/(2*phi), -E/(2*phi), E*phi/2, E*phi/2, -E*phi/2, -E*phi/2])
    icoy = np.array([E/2, E/2, -E/2, -E/2, E/2, E/2, -E/2, -E/2, E/(2*phi), E/(2*phi), -E/(2*phi), -E/(2*phi), E*phi/2, -E*phi/2, E*phi/2, -E*phi/2, 0.0, 0.0, 0.0, 0.0])
    icoz = np.array([E/2, -E/2, E/2, -E/2, E/2, -E/2, E/2, -E/2, E*phi/2, -E*phi/2, E*phi/2, -E*phi/2, 0.0, 0.0, 0.0, 0.0, E/(2*phi), -E/(2*phi), E/(2*phi), -E/(2*phi)])'''

'''pos = [tuple((i,-1,0) for i in np.linspace(-3,3,10))]

print  pos

a = np.linspace(0.0, 10.0, num=10, endpoint=False)
b = np.linspace(10.0, 0.0, num=10)
print np.concatenate((a,b))


print np.array([(x,1,z) for x in np.linspace(0.0, 5.0, 5) for z in np.linspace(0.0, 5.0, 5)])


theta = z/x
r = np.sqrt(x**2+z**2)
x = r*np.cos(theta)
y = r*np.sin(theta)

super = np.array([(r*np.cos(theta), 1, r*np.sin(theta)) for theta in np.linspace(0.0, 2*np.pi, 5, endpoint=False) for r in np.linspace(0.0, 0.5, 5)])

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))\

print super

print unique_rows(super)

p = np.linspace(0, 10, 10)
q = np.linspace(0, 20, 10)
r = np.array((p, q))
swag = np.array([(a, b, 1) for a in p for b in q])
swag2 = np.array([(a, b, 1) for b in q for a in p])

print r
print np.dstack((p,q))

n=1000.0
sigma=1.0

s = np.random.normal(0, sigma, n)
t = np.linspace(0.0, np.pi, n, endpoint=False)

position =  np.resize(np.dstack((s*np.cos(t), np.tile(1,n), s*np.sin(t))), (n,3))

plt.scatter(position[:,0], position[:,2])
plt.show()


print np.dstack((s*t, s, t))'''


'''n=3000
r = np.random.uniform(0, 1, n)
a = np.random.uniform(0, 2*np.pi, n)

x=r*np.cos(a)
y=r*np.sin(a)

x2=np.sqrt(r)*np.cos(a)
y2=np.sqrt(r)*np.sin(a)

points = np.zeros((n,2))

points[:,0]=x2
points[:,1]=y2

pos=points


N = 50
x = np.random.rand(N)
y = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2

fig = plt.figure(figsize=(8, 8))
plt.scatter(pos[:,0],pos[:,1])
plt.show()



fig = plt.figure()
ax = plt.Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')

show()'''

'''x = np.arange(1000)
y = np.arange(1000)+5

plt.hist2d(x, y, bins=40)
plt.colorbar()

plt.savefig('testfig1')
plt.show()

print matplotlib.matplotlib_fname()'''

'''def blocker_mesh(diameter, outer_width, thickness, nsteps=66):
    angles = np.linspace(np.pi/6, 13*np.pi/6, nsteps, endpoint=False)
    s = np.sqrt(3)*(diameter+2*outer_width)
    a = s/(2*np.sqrt(3)*np.cos(np.linspace(np.pi/6, np.pi/2, nsteps/6, endpoint=False)-np.pi/6))
    b = s/(2*np.sqrt(3)*np.cos(5*np.pi/6-np.linspace(np.pi/2, 5*np.pi/6, nsteps/6, endpoint=False)))
    triangle_radii = np.tile(np.concatenate((a, b)), 3)
    #circle = make.linear_extrude(diameter/2*np.cos(angles), diameter/2*np.sin(angles), thickness)
    #triangle = make.linear_extrude(triangle_radii*np.cos(angles), triangle_radii*np.sin(angles), thickness)
   
    x_coords = np.resize(np.dstack((diameter/2*np.cos(angles), triangle_radii*np.cos(angles))), (2*nsteps))
    y_coords = np.resize(np.dstack((diameter/2*np.sin(angles), triangle_radii*np.sin(angles))), (2*nsteps))

    print x_coords
    print y_coords
    print triangle_radii*np.cos(angles)

blocker_mesh(1.0, 0.2, 0.1)

print np.linspace(np.pi/6, -11*np.pi/6, 10)

print np.repeat(2, 10)

k = np.resize(np.tile((1,1,1), 100), (100, 3))
print k

n=5

nx, ny = (2, 25)
x = np.linspace(0, 8, 5)
y = np.linspace(0, 8, 5)
x, y = np.meshgrid(x,y)
print x
print y

print x+y

n= 5.0
x = np.linspace(0.0, 2.0*n-2.0, n)
rep = np.linspace(2.0*n-1.0, 1.0, n)
partial_index = np.linspace(1.0, n-1.0, n-1.0)

xindices = np.repeat(x[0], rep[0])
for i in partial_index:
    xindices = np.concatenate((xindices, np.repeat(x[i],rep[i])))
print xindices

yindices = np.linspace(0.0, 2.0*n-2.0, 2*n-1.0)
for i in partial_index:
    yindices = np.concatenate((yindices, np.linspace(0.0, 2.0*(n-i-1), 2.0*(n-i)-1.0)))
print yindices

xshift = xindices+np.ceil(yindices/2.0)
print xshift

a = np.linspace(4.0, 20.0, 11)
print a
for i in a:
    print i
    print a[-i]'''

'''#hexagonal tiling
n=5
key=np.empty(3*n-2)
for i in np.linspace(0, 3*n-3, 3*n-2):
    key[i] = n-i+2*np.floor(i/3)
xindices = np.linspace(0, 2*n-2, n)
yindices = np.repeat(0,n)
for i in np.linspace(1, 3*n-3, 3*n-3):
    xindices = np.concatenate((xindices, np.linspace(n-key[i], n+key[i]-2, key[i])))
    yindices = np.concatenate((yindices, np.repeat(i,key[i])))

print np.sqrt(3)'''
    
#np.sqrt(3)/2*(E/(2.0*n)*(0.5*x+1)+R-E/(2*np.sqrt(3)))

'''from chroma.transform import normalize
positions = np.empty((3,3))
positions = np.array([[2.0,3.0,4.0], [2.0,5.0,1.0], [0.0,8.0,4.0]])
print positions
print positions[:,0]
for k in np.linspace(0, 2, 3):
    print k
    #print positions[k]'''

'''points = np.empty((20,3))
diameter = 3.0
n = 20
radii = np.linspace(0, diameter/2.0, n)
angles = np.linspace(0, 2*np.pi, n)
points[:,0] = np.sqrt(diameter/2.0)*np.sqrt(radii)*np.cos(angles)
points[:,1] = np.repeat(1.0, n)
points[:,2] = np.sqrt(diameter/2.0)*np.sqrt(radii)*np.sin(angles)

print points
print points[:,0]'''

'''E = 10.0
phi = phi = (1+np.sqrt(5))/2
facecoords = np.array([[E/2, E/2, E/2], [E/2, E/2, -E/2], [E/2, -E/2, E/2], [E/2, -E/2, -E/2], [-E/2, E/2, E/2], [-E/2, E/2, -E/2], [-E/2, -E/2, E/2], [-E/2, -E/2, -E/2], [0.0, E/(2*phi), E*phi/2], [0.0, E/(2*phi), -E*phi/2], [0.0, -E/(2*phi), E*phi/2], [0.0, -E/(2*phi), -E*phi/2], [E/(2*phi), E*phi/2, 0.0], [E/(2*phi), -E*phi/2, 0.0], [-E/(2*phi), E*phi/2, 0.0], [-E/(2*phi), -E*phi/2, 0.0], [E*phi/2, 0.0, E/(2*phi)], [E*phi/2, 0.0, -E/(2*phi)], [-E*phi/2, 0.0, E/(2*phi)], [-E*phi/2, 0.0, -E/(2*phi)]])'''

#view(mesh_from_stl("/home/miladmalek/Dropbox/Chroma/chroma/models/companioncube.stl"))
'''
testarray = np.array([1, 4, 2, 5, 2, 0.2, 4])
print np.amin(testarray)
print np.where([np.amin(testarray)])
print np.argmin(testarray)'''
'''
def infinite_printing(start):
    while True:
        print start
        start += 1
infinite_printing(3)
'''
'''
9, 6, 7, 5, 6

def fbin(pos):
    return np.sum(pos)
pmt = [[] for i in range (10)]
beg = np.array([[0, 0, 0], [2,5,1], [8,4,5], [9, 6, 2], [2, 3, 8], [42, 23, 49]])
end = np.array([[3,4,2], [1,3,2], [1,0,6], [2,3,0], [1,1,4], [6,0,0]])

#begend = np.concatenate((beg, end))
#print begend

for i in range(6):
    bin_ind = fbin(end[i])
    if pmt[bin_ind] == []:
        #print beg[i]
        #print np.vstack((np.vstack((beg[i], beg[i])), beg[i]))
        pmt[bin_ind] = np.expand_dims(beg[i], axis=0)
    else:
        pmt[bin_ind] = np.vstack((pmt[bin_ind], beg[i]))
print pmt
print pmt[6]
#hero = np.empty([1,3])
#hero2 = np.append(hero, np.ones([1,3]), axis = 0)
#print hero2

print pmt[6][1]
print pmt[5]
print pmt[5][0]

print pmt[int(np.floor(5.3))]
'''
'''
sammy = np.zeros((4, 4, 4))

for i in range(2):
    sammy[0,1,2] += 1

print sammy
'''

# pos_array = np.array([[2,3,4],[5,1,7],[2,4,1],[9,4,5]])
# ultra_matrix = np.array([[3,4,5],[1,7,6],[8,8,4]])
# ultra_matrices = np.array([[[3,4,5],[1,7,6],[8,8,4]], [[3,4,5],[1,7,6],[8,8,4]], [[3,4,5],[1,7,6],[8,8,4]]])
# for i in range(np.shape(pos_array)[0]):
#     initposition = np.dot(ultra_matrix, pos_array[i])
#     print initposition

# vector = np.array([2, 3, 4])
# print np.shape(vector)

# print np.sum(ultra_matrices)

# print ultra_matrices/float(np.sum(ultra_matrices))

# something = ultra_matrices
# ultra_matrices = ultra_matrices/2.0
# if np.array_equal(something, ultra_matrices):
#     print "y"
# else:
#     print "n"

# hbar = np.array([3, 4, 2, 1, 7, 5, 4, 6, 2])

# where =  np.where(hbar == 2)[0]

# print np.size(where)

# list1 = [np.array([[1,2],[3,4]]), np.array([[5, 6], [7,8]])]
# list2 = [np.array([[1,2],[3,4]]), np.array([[5, 6], [7,8]])]

# print np.array_equal(list1, list2)

# firstarray = np.array([2, 3, 4, 2, 5])
# for i in range(1000000):
#     secondarray = np.concatenate((firstarray, np.array([2, 4])))
# print secondarray

# choices = np.array([2, 5, 6, 9])
# stupidarray = np.array([2,3,4,5,6,3,5,1,1,4,6.4,6,3])
# betterarray = np.empty(np.size(choices))
# for i in range(np.size(choices)):
#     happy = choices[i]
#     betterarray[i] = stupidarray[happy]
# print betterarray
    
# class Test(object):
#     def __init__(self, by):
#         self.answer = by*5
#         self.n = 34
#         print answer

# Test(3)

# A = np.array([[2,3,4], [4,7,1], [9,0,3]])

# B = np.array([[1,2,0], [9,4,6], [3,6,4]])
# AB = np.multiply(A, B)
# print AB

# print np.nonzero(AB)
# print np.nonzero(AB)[0]

# print AB[2,0]


# print 1e-323

# from kabamland import find_inscribed_radius
# print find_inscribed_radius(10)

# x = [i for i in range(10)]
# print x

# print x[9]

# lizard = np.empty((10,10,10))

# monkey = np.zeros((10,10,10))

# print lizard[2,2,2]

# lplus1 = lizard[2,2,2] = lizard[2,2,2]+1

# print lizard[2,2,2]

# length = 10
# facebin_array = np.empty(length)
# pmt_positions = np.empty((length, 3))
# k = 4

# initialcoords = np.array([[0,0,3], [1,1,4], [2,2,0], [3,3,3], [4,4,4], [5,5,1], [6,6,0], [7,7,2], [8,8,4], [9,9,0]])
# initialzcoord = np.array([3, 4, 0, 3, 4, 1, 0, 2, 4, 0])
# boolean_array = (initialzcoord < 1e-5) & (initialzcoord > -1e-5) 

# bool2 = np.column_stack((boolean_array, boolean_array, boolean_array))
# bool3 = np.reshape(np.repeat(boolean_array, 3), (10, 3))
# print bool2

# np.place(facebin_array, boolean_array, k)
# np.copyto(pmt_positions, initialcoords, where=bool3)
# print pmt_positions

# #print facebin_array

#print -np.ones(5).astype(int)

# dwayne = np.linspace(1,10,5)
# print dwayne

# def dosomething(input1, input2):
#     ball = dwayne * input1 - input2
#     return ball

# print dosomething(3,2)

# a = np.array([[[0,0,0], [1,1,1], [0,0,0]], [[1,1,1], [3,3,3], [1,1,1]], [[0,0,0], [2,2,2], [0,0,0]]])

# x, y, z = np.mgrid[0:4, 0:4, 0:4]
# bin_array = zip(x.ravel(), y.ravel(), z.ravel())
# event_pdf = np.ones((10, 10, 10))
# # for i in bin_array:
# #     print type(i)

# neighbors = np.ones((6, 3))
# yo = tuple(neighbors[0])
# print yo
# print a[yo]

# shape = np.shape(a)
# x, y, z = shape[0], shape[1], shape[2]

# print x, y, z

# b = np.array(([0,0,0,0],
# [0,10,10,0],
# [0,10,10,0],
# [0,10,10,0]))

# print a[2,1,0]

# print np.sum(b)

# com = ndimage.measurements.center_of_mass(b)
# print com
# print np.round(com)
# print np.round(np.array([6.1, 6.1, 6.1]))
# print np.unravel_index(np.argmax(b), (4,4))

# def bin_to_position(bintup, detectorxbins, detectorybins, detectorzbins):
#     #takes a bin tuple and outputs the coordinate position tuple at the CENTER of the bin
#     #origin is at center of whole configuration
#     inscribed_radius = 7.58
#     xpos = (bintup[0]+(1.0-detectorxbins)/2.0)*(2.0*inscribed_radius/detectorxbins)
#     ypos = (bintup[1]+(1.0-detectorybins)/2.0)*(2.0*inscribed_radius/detectorybins)
#     zpos = (bintup[2]+(1.0-detectorzbins)/2.0)*(2.0*inscribed_radius/detectorzbins)
#     return (xpos,ypos,zpos)

# print bin_to_position((9, 8, 7), 10, 10, 10)

# # x, y = np.mgrid[0:5, 0:5]
# x = (x + (1.0-5)/2.0)*(2.0*10.0/5)
# y = (y + (1.0-5)/2.0)*(2.0*10.0/5)

# points2d = zip(x.ravel(), y.ravel())

# x, y, z = np.mgrid[0:4, 0:4, 0:4]
# bin_array = zip(x.ravel(), y.ravel(), z.ravel())
# print np.shape(bin_array)
# print bin_array[26], bin_array[46]
# #print bin_array[42]
# x = (x + (1.0-4)/2.0)*(2.0*8.0/4)
# y = (y + (1.0-4)/2.0)*(2.0*8.0/4)
# z = (z + (1.0-4)/2.0)*(2.0*8.0/4)
# points = zip(x.ravel(), y.ravel(), z.ravel())
# from scipy import spatial

# #print points
# print points[42]

# tree = spatial.KDTree(points)
# neighbors= tree.query_ball_point([2.0, 2.0, 2.0], 4, eps=1e-5)
# print neighbors
# for i in range(len(neighbors)):
#     #print points[neighbors[i]]
#     print bin_array[neighbors[i]]

# final_pdf = np.ones((10, 10, 10))

# print final_pdf[5,3,2]

# radii = np.linspace(0, 7.5, 10)
# radii[0] = 1e-7
# print radii

# print np.arange(10)
# x, y, z = np.mgrid[0:4, 0:4, 0:4]
# print x.ravel()
# print y.ravel()
# print z.ravel()

# probs = np.zeros(10)
# vals = np.linspace(1, 10, 10)
# for i in range(10):
#     for j in range(len(vals)):
#         probs[i] = probs[i] + vals[j]
# print probs
# print vals
# print np.sum(vals)

# uno = np.array([0, 1, 2, 3, 4])
# dos = np.array([23, 45, 59, 94, 134])
# d = np.dstack((uno, dos))
# print d
# print np.hstack((uno, dos))
# print np.vstack((uno, dos))
# print np.shape(d)
# print np.reshape(d, (5, 2))

#print np.round(6.1, 6.1, 6.1)


# print type((3, 3, 3))


# easy_array = np.array([[2,0,2]])
# find_max_starting_zone(easy_array)


        
        
# print np.sum(np.array([-1,-1,-1]))
        
# print 5 != 6-1

# print 7.0-2

# print 3 == 3.0

# neighbors = np.array([[2,3,4],[-1,-1,-1],[2,3,2]])
# print neighbors[2]

# first = (1,1,2)
# center = np.array(first)*.002
# center += np.array([1,1,1])*.009
# print center/.011

# print 45.5/5

# print float(0.0000002)

# hi = np.array([[1,1,1],[2,2,2], [3,3,3]])
# second = np.array([5,6,7])
# for i in range(3):
#     second +=  hi[i]*2
#     print second
# print second


# def gaussint(radius, sigma):
#         return np.sqrt(2/np.pi)/sigma**3*integrate.quad(lambda x: x**2*np.exp(-x**2/(2.0*sigma**2)), 0, radius)[0]

# def testfunction(radius, sigma):
#         return integrate.quad(lambda x: x, 0, radius)

# print gaussint(1, 1)
# print gaussint(2, 1)
# print gaussint(3, 1)

# print testfunction(1, 1)
# print testfunction(2, 1)
# print testfunction(3, 1)

def find_max_radius(edge_length, base):
    #finds the maximum radius of a lens
    max_radius = edge_length/(2*np.sqrt(3)*base)
    return max_radius

#print find_max_radius(10, 9)/find_max_radius(10, 6)
#print find_max_radius(10, 6)

def find_focal_length(edge_length, base, diameter_ratio, thickness_ratio):
    max_radius = find_max_radius(edge_length, base)

    #focuses a ray of light from radius x along the parabolic lens to 'focal_length'
    outside_refractive_index = lm.ls_refractive_index
    inside_refractive_index = lm.lensmat_refractive_index
    diameter = 2*max_radius*diameter_ratio
    thickness = diameter*thickness_ratio
    x = diameter/4.0
    
    #calculation of focal_length
    a = 2.0*thickness/diameter**2
    H = a/4*diameter**2
    u = np.arctan(2*a*x)
    m = -np.tan(np.pi/2-u+np.arcsin(outside_refractive_index/inside_refractive_index*np.sin(u)))
    b = a*x**2-m*x-H
    X = (m+np.sqrt(m**2+4*a*(-a*x**2+m*x+2*H)))/(-2*a)
    p = np.arctan((2*a*m*X-1)/(2*a*X+m))
    q = np.arcsin(inside_refractive_index/outside_refractive_index*np.sin(p))
    M = (1+2*a*X*np.tan(q))/(2*a*X-np.tan(q))
    focal_length = -a*X**2+H-M*X
    return focal_length
#print find_focal_length(10.0, 6, 0.5, 0.25)
#print find_focal_length(10.0, 9, 0.75, 0.25)

def find_thickness_ratio(edge_length, base, diameter_ratio, focal_length):
    #Finds the thickness ratio to make a given lens have a particular focal length
    def F(thickness_ratio):
        return (find_focal_length(edge_length, base, diameter_ratio, thickness_ratio) - focal_length)**2
    
    happy = optimize.fmin(F, 0.001, xtol=1e-6)
    #starting off with a low initial guess takes longer but gets the correct answer more of the time.
    #print happy
#find_thickness_ratio(10, 9, 0.5, find_focal_length(10.0, 9, 0.5, 0.25))


#def find_thickness_ratio2(edge_length, base, diameter_ratio, focal_length):
    #Finds the thickness ratio to make a given lens have a particular focal length
 #    def F(thickness_ratio):
#         return (find_focal_length(edge_length, base, diameter_ratio, thickness_ratio) -focal_length)**2
   
#     minimum = 0.1
#     minimum_value = F(minimum)
#     for i in np.linspace(0.1, 1, 10):
#         if F(i) < minimum_value:
#             minimum = i
#             minimum_value = F(i)
#     for i in np.linspace(i-0.1, i+0.1, 11):
#         if i == 0:
#             continue
#         elif F(i) < minimum_value:
#             minimum = i
#             minimum_value = F(i)

    
# # find_thickness_ratio2(10, 9, 0.5, 0.2049)

# print np.linspace(0.2, 0.4, 11)

# angles = np.linspace(0, np.pi/2, 10, endpoint=False)
# print angles


# print np.tile((-5, 2, 0), 10)

# def my_fun(num):
#     return num*2

# print my_fun(4)

# R1 = 1.5
# R2 = -0.2
# diameter = 1
# if (abs(R1) or abs(R2)) < diameter/2.0:
#     print 'no'
# print (abs(R1) or abs(R2)) < diameter/2.0

# print (abs(R1) or abs(R2))

# listA = np.array([-4, -2, 2, 4])
# listB = np.array([5, 6, 7, 8])
# print listA + listB
# print np.add(listA, listB)
# print np.add(listA**2, listB**2)
# print np.sqrt(np.add(listA**2, listB**2))
# print np.std(np.sqrt(np.add(listA**2, listB**2)))

# listC = np.array([-3, -2, -1,  1, 2, 3])
# listD = np.array([3, 2, 1,  1, 2, 3])
# print np.std(listC), np.std(listD)

# listE = np.linspace(-1, 1, 10000)
# print np.std(listE)
# print np.sqrt(1.0/12)

# ghost = np.array([[1, 2, 3], [2, 3, 4], [1, 1, 0], [9, 5, 6]])
# miracle = np.array([[2, 9, 3], [0, 5, 1], [1, 2, 4], [7, 6, 0]])


# print ghost
# print ghost[np.array([1, 3])]

# num = np.shape(ghost)[0]
# print np.mean(ghost, 1)
# print 13/4.0, 11/4.0

# angles = np.array([1, 1, 0])
# print np.concatenate((np.array([0]), angles))

# print np.zeros(1)
# print np.array([0])

# print np.argmin(angles)


# print angles/np.linalg.norm(angles)

# print normalize(angles)

# print ghost/np.linalg.norm(ghost)
# print normalize(ghost)

# angles = np.array([3, 4, 5, 2, 3, 6, 3, 12, -1, 5, 6, -1, 3, 8, 4, 5, 6, 4])
# angles2 = np.array([2, 4, 2, 4])
# print np.append(angles, angles2)
# angles = np.sort(angles)
# print angles
# firstgood = 0
# length = np.shape(angles)[0]
# for i in range(length):
#     if angles[i] == -1:
#         continue
#     else:
#         firstgood = i
#         break
# print angles[firstgood:]

# new_angles = np.array([1, 1, 1.2, 1.5, 2, 3, 3.4, 3.5, 3.9,  4, 4.1, 4.9, 5, 6.7, 9, 10, 11, 11.6, 12])
# bestest = np.linspace(0, 100, 101, endpoint=True)


# def choose_values(values, num):
#     #choose num amount of points chosen evenly through the array values- not including endpoints. In essence the particular method used here creates num bins and then chooses the value at the center of each bin, rounding down if it lands between two values.
#     length = np.shape(values)[0]
#     half_bin_size = (length-1.0)/(2.0*num)
#     odd_numbers = np.linspace(1, 2*num-1, num)
#     indices = (odd_numbers*half_bin_size).astype(int)
#     chosen_values = np.take(values, indices)
#     return chosen_values, length
        
# #print choose_values(new_angles, 3)
# print choose_values(new_angles, 3)[0]

# normdir = (-0.3333, -0.9428, 0)
# normdir = np.array(normdir)
# theta = np.arccos(np.dot(normdir, (0, -1, 0)))
# print theta
# print np.degrees(theta)
# theta2 = np.arcsin(1.5/2.0*np.sin(theta))
# print theta2
# print np.degrees(theta2)
# print np.cos(theta2)

# print 'New'
# base = 4
# edge_length = 10.0
# max_radius = 1.0

# key = np.empty(3*base-2)
# for i in range(3*base-2):
#     key[i] = base-i+2*np.floor(i/3)
# xindices = np.linspace(0, 2*(base-1), base)
# yindices = np.repeat(0,base)
# for i in np.linspace(1, 3*(base-1), 3*(base-1)):
#     xindices = np.concatenate((xindices, np.linspace(base-key[i], base+key[i]-2, key[i])))
#     yindices = np.concatenate((yindices, np.repeat(i,key[i])))
# xcoords = edge_length/(2.0*base)*(xindices+1)-edge_length/2.0
# ycoords = max_radius*(yindices+1)-edge_length/(2*np.sqrt(3))

# print xindices 
# print yindices
# print xcoords
# print ycoords

# xindices = np.linspace(0, 2*(base-1), base)
# yindices = np.repeat(0, base)
# for i in np.linspace(1, base-1, base-1):
#     xindices = np.append(xindices, np.linspace(i, 2*(base-1)-i, base-i))
#     yindices = np.append(yindices, np.repeat(i, base-i))
# xcoords = 
# print xindices

import numpy as np

# def find_nearest_vector(array, value):
#   idx = np.array([np.linalg.norm(x+y+z) for (x,y,z) in array-value]).argmin()
#   return array[idx]

# A = np.random.random((10,3))*100
# pt = [6, 30, 20]
# print A  
# print find_nearest_vector(A,pt)


direction_array = np.array(((2, 3, 4), (1, 2, 3), (2, 4, 3), (2, 4, 1)))
second = np.array(((1, 1, 1), (3, 3, 3), (1, 1, 1), (2, 1, 0)))
pmt_bins = np.array([3, 2, 3, 4])

##calibrate
pmt_indices = np.where(pmt_bins == 3)[0]
print direction_array[pmt_indices]
norms = np.repeat(1.0, 4)
end_direction_array = normalize(direction_array)
mean_angle = normalize(np.mean(end_direction_array, axis=0))
projection_norms = np.dot(end_direction_array, mean_angle)
orthogonal_complements = np.sqrt(norms**2 - projection_norms**2)
error = np.std(orthogonal_complements, ddof=1)
print mean_angle

#print np.tensordot(direction_array, second, axes=(1,0))
# print direction_array - second
# detectorbin_array = np.array((3, 2, 2, 3))
# print detectorbin_array
# hi = np.where(detectorbin_array == 3)
# print hi
# print direction_array[hi]
# print np.mean(direction_array[hi], axis=0)
# stacked_array = np.vstack((direction_array, second))
# print stacked_array
# print np.shape(stacked_array)
# appended_array = np.append(direction_array, second)
# print appended_array
# print np.reshape(appended_array, (8,3))


# print np.append(direction_array, second, axis=0)

# new_array = -1
# if new_array == -1:
#     new_array = np.array(((1, 1, 1), (2, 1, 2)))
#     second_array = np.array(((2,3,4)))

# best_array = np.vstack((new_array, second_array))
# print best_array
# tree = spatial.KDTree(best_array)
# end_dir = np.array(((2.1, 3, 3.9), (0.9, 1, 1)))
# print tree.query(end_dir)


# xbin = 7
# ybin = 47 
# zbin = 33

# xbins = 11
# ybins = 82
# zbins = 92

# detbin = xbin + ybin*xbins + zbin*xbins*ybins

# print detbin

# xbin2 = (detbin % (xbins*ybins)) % xbins

# ybin2 = ((detbin-xbin2) % (xbins*ybins))/xbins

# zbin2 = (detbin - xbin2 - ybin2*xbins)/(xbins*ybins)

# print xbin2, ybin2, zbin2

# full_pmt_bin_array = np.array((3, 4, 1, 2, 5, 3, 2, 3))
# full_detector_bin_array = np.array((27, 23, 24, 25, 23, 29, 28, 27))
# pmt_indices = np.where(full_pmt_bin_array == 3)[0]
# detectorbins_for_pmt = full_detector_bin_array[pmt_indices]
# detector_indices = np.where(detectorbins_for_pmt == 27)[0]
# direction_indices = pmt_indices[detector_indices]
# print pmt_indices
# print detectorbins_for_pmt
# print detector_indices
# print direction_indices


# def bin_to_position_array(detectorxbins, detectorybins, detectorzbins):
#     #takes a shape for the detector bins and returns an array of coordinate centers for each bin.
#     x, y, z = np.mgrid[0:detectorxbins, 0:detectorybins, 0:detectorzbins]
#     xpos = (x+(1.0-detectorxbins)/2.0)*(2.0*7.5/detectorxbins)
#     ypos = (y+(1.0-detectorybins)/2.0)*(2.0*7.5/detectorybins)
#     zpos = (z+(1.0-detectorzbins)/2.0)*(2.0*7.5/detectorzbins)
#     position_array = zip(xpos.ravel(), ypos.ravel(), zpos.ravel())
#     return position_array

# pos_array = bin_to_position_array(10, 10, 10)
# print np.shape(pos_array)


#pmt bin to facebin, xbin, ybin:
# xbins = 68
# ybins = 35
# def pmt_bin(facebin, xbin, ybin):
#     pmtbin = facebin*(xbins*ybins) + ybin*xbins + xbin
#     return pmtbin

# #print pmt_bin(0, 8, 4)
# my_bin = pmt_bin(8, 12, 21)
# print my_bin

# def pmtbin_to_tuple(pmtbin, xbins, ybins):
#     #takes a pmtbin number and returns the tuple of its facebin, xbin, and ybin.
#     xbin = pmtbin % xbins
#     ybin = ((pmtbin-xbin)/xbins) % ybins
#     facebin = (pmtbin-ybin*xbins-xbin)/(xbins*ybins)
#     return (facebin, xbin, ybin)

# print pmtbin_to_bins(my_bin, 68, 35)
# print type(pmtbin_to_bins(my_bin, 68, 35))

# array = np.array(([2], [3], [1], [5], [3]))
# print array
# print (array + 1)/2.0

# print np.std([1,2,3,4])
# print np.var([1,2,3,4])
# print np.sqrt(1.25)

#import numpy.ma as ma
total_means = np.random.randint(5, size=(3, 10, 3))
amount_of_hits = np.random.randint(10, size=(3, 10))
total_variances = np.random.random((3, 10))
#total_variances[:,2] = 0
amount_of_hits[:,2] = 0
amount_of_hits[:,1] = 1
#print total_means
print 'hi'
print total_variances
print amount_of_hits
print amount_of_hits-1
reshaped_amount_of_hits = np.reshape(np.repeat(amount_of_hits, 3), (3, 10, 3))
#print reshaped_amount_of_hits

summed_hits = np.sum(amount_of_hits, axis=0)
bad_pmts = np.where(summed_hits <= 3)[0]
print 'bad_pmts', bad_pmts

m_means = np.zeros_like(total_means)
m_means[:, bad_pmts] = 1
m_variances = np.zeros_like(total_variances)
m_variances[:, bad_pmts] = 1

masked_total_means = np.ma.masked_array(total_means, m_means)
masked_total_variances = np.ma.masked_array(total_variances, m_variances)
print masked_total_variances

averaged_means = np.ma.average(masked_total_means, axis=0, weights=reshaped_amount_of_hits)

averaged_variances = np.ma.average(masked_total_variances, axis=0, weights=amount_of_hits-1)

print 'averaged_means'
print averaged_means

print 'averaged_variances'
print averaged_variances

averaged_variances.mask = np.ma.nomask
averaged_means.mask = np.ma.nomask

print averaged_means
print averaged_variances 
