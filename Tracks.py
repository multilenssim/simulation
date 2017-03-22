import numpy as np

class Tracks(object):
    '''A Tracks object contains a list of photon tracks and their uncertainties, working
    backward from their point of detection. The current implementation represents them as
    a hit position, a direction (in 3D), and an uncertainty on the transverse position,
    normalized to unit length (so the track profile is a cone).
    '''
    def __init__(self, hit_pos, means, sigmas):
        self.hit_pos = hit_pos # (3, n) numpy array
        self.means = means # (3, n) numpy array
        self.sigmas = sigmas # (n,) numpy array
        
    def __len__(self):
        #Returns the number of tracks in self.
        return len(self.sigmas)
        
    def __iter__(self):
        #Allows for iterating over tracks. 
        #Returns results as a tuple (hit_pos, mean, sigma).
        for ii in range(len(self)):
            yield (self.hit_pos[:,ii], self.means[:,ii], self.sigmas[ii])
        
    def cull(self, ind_remain):
        # Reduces tracks to only those with indices in ind_remain
        ind_remain = np.unique(ind_remain)
        self.hit_pos = self.hit_pos[:,ind_remain]
        self.means = self.means[:,ind_remain]
        self.sigmas = self.sigmas[ind_remain]
        
    def closest_pts_sigmas(self, v):
        # Returns an array of positions along the tracks closest to Vertex v, along with
        # an array of the sigmas scaled by the distance along the track to that point
        r_vec = (v.pos - self.hit_pos.T).T
        r_proj = np.einsum('ij,ij->j',r_vec,self.means) # Get projections onto direction vectors
        #print "r_proj: " + str(r_proj)
        r_fin = self.hit_pos+self.means*r_proj
        sig_scaled = self.sigmas*r_proj
        return r_fin, sig_scaled
        
        
class Vertex(object):
    '''A Vertex object contains an estimated photon source location, along with
    its energy (in number of photons) and position uncertainty.
    '''
    def __init__(self, pos, err, n_ph):
        self.pos = pos # (3,) numpy array
        self.err = err # (3,) numpy array
        self.n_ph = n_ph # float
        
    def dist(self, r):
        # Returns distances from vertex to array of positions, r, with shape (3, n)
        return np.linalg.norm(r.T-self.pos,axis=1)
