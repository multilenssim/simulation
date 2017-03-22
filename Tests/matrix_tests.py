import numpy as np

w = np.random.rand(2)
v = np.random.rand(3,1)
r = np.random.rand(3,2)

a = np.random.rand(3,2)
d = np.random.rand(2)
print np.shape(d)

a1 = np.einsum('...j,...ij->...ij',v,r)
print a1

test = v-r
print test
print np.shape(test)
print np.shape(d)
print test/d
print d

print 'tensor'
print np.tensordot(a.T,r,1) # Sum over first (one) axis of a, last (one) axis of r
print 'matrix'
print (a.T).dot(r)
print 'einsum'
print np.einsum('ji,ji->i',a,r)

print 'loop'

a2 = np.zeros(2)
for ii in range(2):
    for jj in range(3):
        #print v[jj],r[ii,jj]
        a2[ii] += a[jj,ii]*r[jj,ii]

print a2        

print 1/(1+np.exp(v**2/2.0))

print "a'"
print w/d
print a.T
print w*a.T/d