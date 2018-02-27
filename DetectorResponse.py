#from ShortIO.root_short import PDFRootWriter, PDFRootReader, ShortRootReader, AngleRootReader, AngleRootWriter
from kabamland2 import get_curved_surf_triangle_centers, get_lens_triangle_centers
from chroma.transform import make_rotation_matrix, normalize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#import lensmaterials as lm
from scipy import spatial 
import detectorconfig
import numpy as np
#import time


class DetectorResponse(object):
    '''A DetectorResponse represents the information available to the detector
    after calibration. There are multiple subclasses, allowing for multiple
    representations of this information, but essentially it boils down to
    knowing where a photon hitting any particular PMT is likely to have come from.
    The configuration of the detector is also stored in this object, so that
    its geometry is known.    
    '''
    def __init__(self, configname, detectorxbins=10, detectorybins=10, detectorzbins=10):
        config = detectorconfig.configdict(configname)
        self.configname = configname  # Adding this for intermediate calibration file writing
        self.is_calibrated = False
        self.lns_rad = config.half_EPD/config.EPD_ratio
        self.detectorxbins = detectorxbins
        self.detectorybins = detectorybins
        self.detectorzbins = detectorzbins
        #self.edge_length, self.facecoords, self.direction, self.axis, self.angle, self.spin_angle = return_values(config.edge_length, config.base)
        self.pmtxbins = config.pmtxbins
        self.pmtybins = config.pmtybins
        self.n_lens_sys = config.base # Number of lens systems per face
        self.detector_r = config.detector_r
        self.nsteps = config.nsteps

        self.n_triangles_per_surf = int(2*self.nsteps*int((self.nsteps-2)/2.))
        self.n_pmts_per_surf = int(self.n_triangles_per_surf/2.)

        #if not self.detector_r:
        #    self.npmt_bins = 20*self.pmtxbins*self.pmtybins
        #else:
        #    self.npmt_bins = 20*self.n_lens_sys*self.n_pmts_per_surf # One curved detecting surf for each lens system
        
        self.diameter_ratio = config.diameter_ratio
        self.thickness_ratio = config.thickness_ratio
        ##changed
        self.focal_length = config.focal_length

        ##end changed
        #self.pmt_side_length = np.sqrt(3)*(3-np.sqrt(5))*self.focal_length
        self.inscribed_radius = config.edge_length
        #self.rotation_matrices = self.build_rotation_matrices()
        #self.inverse_rotation_matrices = np.linalg.inv(self.rotation_matrices)
        #self.displacement_matrix = self.build_displacement_matrix()
        #self.inverse_rotated_displacement_matrix = self.build_inverse_rotated_displacement_matrix()
        #self.lens_inverse_rotated_displacement_matrix = self.build_lensplane_inverse_rotated_displacement_matrix()
        #new properties for curved surface detectors

        # Temporarily comment out to allow access to old calibration files
        self.triangle_centers,self.n_triangles_per_surf,self.ring = get_curved_surf_triangle_centers(config.vtx, self.lns_rad, self.detector_r, self.focal_length, self.nsteps, config.b_pixel)
        self.triangle_centers_tree = spatial.cKDTree(self.triangle_centers)
        self.n_pmts_per_surf = int(self.n_triangles_per_surf/2.)

        if not self.detector_r:
            self.npmt_bins = 20*self.pmtxbins*self.pmtybins
        else:
            self.npmt_bins = self.n_lens_sys*self.n_pmts_per_surf # One curved detecting surf for each lens system

        # Temporarily comment out to allow access to old calibration files
        self.lens_centers = get_lens_triangle_centers(config.vtx, self.lns_rad, config.diameter_ratio, config.thickness_ratio, config.half_EPD, config.blockers, blocker_thickness_ratio=config.blocker_thickness_ratio, light_confinement=config.light_confinement, focal_length=config.focal_length, lens_system_name=config.lens_system_name)
        self.lens_rad = config.half_EPD 

        #self.calc1 = self.pmtxbins/self.pmt_side_length
        #self.calc2 = self.pmtxbins/2.0
        #self.calc3 = 2*self.pmtybins/(np.sqrt(3)*self.pmt_side_length)
        #self.calc4 = self.pmtybins/3.0
        #self.calc5 = self.pmtxbins*self.pmtybins
        self.c_rings = np.cumsum(self.ring)
        self.c_rings_rolled = np.roll(self.c_rings, 1)
        self.c_rings_rolled[0] = 0


    def build_rotation_matrices(self):
        rotation_matrices = np.empty((20, 3, 3))
        for k in range(20):
            rotation_matrices[k] = np.dot(make_rotation_matrix(self.spin_angle[k], self.direction[k]), make_rotation_matrix(self.angle[k], self.axis[k]))
        return rotation_matrices

    def build_displacement_matrix(self):
        displacement_matrix = np.empty((20, 3))
        for k in range(20):
            displacement_matrix[k] = self.facecoords[k] + self.focal_length*normalize(self.facecoords[k])
        return displacement_matrix

    def build_inverse_rotated_displacement_matrix(self):
        inverse_rotated_displacement = np.empty((20, 3))
        for k in range(20):
            inverse_rotated_displacement[k] = np.dot(self.inverse_rotation_matrices[k], self.displacement_matrix[k])
        return inverse_rotated_displacement

    def build_lensplane_inverse_rotated_displacement_matrix(self):
        lens_inverse_rotated_displacement = np.empty((20, 3))
        for k in range(20):
            lens_inverse_rotated_displacement[k] = np.dot(self.inverse_rotation_matrices[k], self.facecoords[k])
        return lens_inverse_rotated_displacement

    def calibrate(self, filename, nevents=-1):
        # Use with a simulation file 'filename' to calibrate the detector
        self.is_calibrated = True
        print "Base class DetectorResponse has no specific calibration method - instantiate as a subclass."
        
    def angles_response(self, config, simname, nolens=False, rmax_frac=1.0):
        #takes a simulation file and creates an array of angles that photons hit the pmts at.
        # (replace lenses with disk pmts for the simulation to see what angles light hits the lenses at.
        # Light needs to land on an icosahedron face plane so use disks instead of just changing the surface of the lens to detecting.)
        # ev.photons_end.dir[detected] is always 0s as of now because we aren't saving directions in event or simulation files.

        #If nolens is True, assumes a "perfectres" type detector, with no lenses instead of lenses replaced with PMTs.
        #Restricts starting photons to be w/in rmax_frac*inscribed_radius of the center.
        reader = ShortRootReader(simname)
        total_angles = np.zeros(0)
        loops = 0
        for ev in reader:
            detected = (ev.photons_end.flags & (0x1 <<2)).astype(bool)
            # Check which photons start w/in r_max of center
            r_max = rmax_frac*self.inscribed_radius
            r0 = np.linalg.norm(ev.photons_beg.pos, axis=1)
            #print r0
            start_in_bounds = r0 < r_max
            #print start_in_bounds
                     
            use_photon = np.logical_and(detected, start_in_bounds)          
            beginning_photons = ev.photons_beg.pos[use_photon]
            ending_photons = ev.photons_end.pos[use_photon]
            transpose_pos_array = np.transpose(ending_photons)
            length = np.shape(ending_photons)[0]
 
            facebin_array = -np.ones(length).astype(int)
            # finding which LENS face each position belongs to by seeing which inverse rotation and displacement brings the position closest to z = 0. Note this is different than finding the PMT face as in find_pmt_bin_array.
            # Will find the PMT face if nolens is True
            for k in range(20):
                if nolens:
                    initial_position_array = np.transpose(np.dot(self.inverse_rotation_matrices[k], transpose_pos_array)) - self.inverse_rotated_displacement_matrix[k]
                else:
                    initial_position_array = np.transpose(np.dot(self.inverse_rotation_matrices[k], transpose_pos_array)) - self.lens_inverse_rotated_displacement_matrix[k]

                wherezeros = (initial_position_array[:,2] < 1e-5) & (initial_position_array[:,2] > -1e-5)
                np.place(facebin_array, wherezeros, k)
            #print facebin_array

            #finding the angles between each direction and facecoord (facecoords[k] representing the optical axis of every lens on the kth face.)
            directions = normalize(ending_photons - beginning_photons)
            face_directions = normalize(self.facecoords)
            angles = -np.ones(length)
            for i in range(length):
                face = facebin_array[i]
                if face == -1:
                    continue
                angles[i] = np.arccos(np.dot(directions[i], face_directions[face]))

            if total_angles == []:
                total_angles = angles
            else:
                total_angles = np.append(total_angles, angles)

            loops += 1
            if np.mod(loops, 10) == 0:
                print 'loop: ' + str(loops)
            if loops == 100:
                break

        #sorting from lowest to highest and removing any -1s. 
        total_angles = np.sort(total_angles)
        first_good = 0
        for i in range(np.shape(total_angles)[0]):
            if total_angles[i] == -1:
                continue
            else:
                first_good = i
                break
        total_angles = total_angles[first_good:]
        
        def choose_values(values, num):
        #choose num amount of points chosen evenly through the array values- not including endpoints. In essence the particular method used here creates num bins and then chooses the value at the center of each bin, rounding down if it lands between two values.
            length = np.shape(values)[0]
            half_bin_size = (length-1.0)/(2.0*num)
            odd_numbers = np.linspace(1, 2*num-1, num)
            indices = (odd_numbers*half_bin_size).astype(int)
            chosen_values = np.take(values, indices)
            return chosen_values

        my_values = choose_values(total_angles, 6)

        print 'length: ' + str(np.shape(total_angles)[0])
        print 'first_good: ' + str(first_good)
        print 'total_angles: ' + str(total_angles)
        print 'chosen_values: ' + str(my_values)
        print 'chosen_values in degrees: ' + str(np.degrees(my_values))

        fig = plt.figure(figsize=(7.8, 6))
        plt.hist(total_angles, bins=100)
        plt.xlabel('Angles')
        plt.ylabel('Amount')
        plt.title('Angles Response Histogram for ' + config)
        plt.show()

        return total_angles

    def build_perfect_resolution_direction_list(self, simname, filename):
        # creates a list of average ending direction per detectorbin per pmtbin. Each pmtbin will have an array of detectorbins with an average ending direction of photons eminating from each one for each one. Simulation must just be the pmticosohedron.  Make sure that the focal length is the same for the simulation as it is in this file at the top.
        reader = ShortRootReader(simname)
        detectorbins = self.detectorxbins*self.detectorybins*self.detectorzbins
        calc6x = self.detectorxbins/(2*self.inscribed_radius)
        calc6y = self.detectorybins/(2*self.inscribed_radius)
        calc6z = self.detectorzbins/(2*self.inscribed_radius)
        calc7x = self.detectorxbins/2.0
        calc7y = self.detectorybins/2.0
        calc7z = self.detectorzbins/2.0

        culprit_count = 0
        loops = 0        
        full_length = 0
        for ev in reader:
            detected = (ev.photons_end.flags & (0x1 <<2)).astype(bool)
            beginning_photons = ev.photons_beg.pos[detected]
            ending_photons = ev.photons_end.pos[detected]
            end_direction_array = normalize(ending_photons-beginning_photons)
            length = np.shape(ending_photons)[0]
            full_length += length

          #creating an array of pmt bins for the ending_photons
            pmt_bin_array = self.find_pmt_bin_array(ending_photons)

            # creating arrays for the detector bins of each photon.
            xinitbin_array = np.floor(calc6x*beginning_photons[:,0]+calc7x).astype(int)
            yinitbin_array = np.floor(calc6y*beginning_photons[:,1]+calc7y).astype(int)
            zinitbin_array = np.floor(calc6z*beginning_photons[:,2]+calc7z).astype(int)
            # creating a single bin for each detectorbin.
            detector_bin_array = xinitbin_array + yinitbin_array*self.detectorxbins + zinitbin_array*self.detectorxbins*self.detectorybins

            if loops == 0:
                full_end_direction_array = end_direction_array
                full_pmt_bin_array = pmt_bin_array
                full_detector_bin_array = detector_bin_array
            else:
                full_end_direction_array = np.append(full_end_direction_array, end_direction_array)
                full_pmt_bin_array = np.append(full_pmt_bin_array, pmt_bin_array)
                full_detector_bin_array = np.append(full_detector_bin_array, detector_bin_array)

            loops += 1
            if loops == 50:
                break
            print loops

        full_end_direction_array = np.reshape(full_end_direction_array, (full_length, 3))
        print 'number of hits: ' + str(full_length)

        detector_dir_list = [[] for i in range(self.npmt_bins)]
        #for each of the entries in detector_dir_list we want an array of length detectorbins where in each slot of each array we have the average ending direction corresponding to that detector (for photons from that detector going to the pmt cooresponding to the entry index of the whole list.) This makes an array of shape: (detxbins*detybins*detzbins, 3).     
   
        for i in range(self.npmt_bins):
            if i % 10000 == 0:
                print str(i) + ' out of ' + str(self.npmt_bins)
            pmt_indices = np.where(full_pmt_bin_array == i)[0]
            if np.shape(pmt_indices)[0] == 0:
                culprit_count += 1
                continue
            detectorbins_for_pmt = full_detector_bin_array[pmt_indices]
            average_direction_array = -5*np.ones((detectorbins, 3)).astype(np.float32)
                
            for j in range(detectorbins):
                detector_indices = np.where(detectorbins_for_pmt == j)[0]
                if np.shape(detector_indices)[0] == 0:
                    continue
                direction_indices = pmt_indices[detector_indices]
                ending_directions = full_end_direction_array[direction_indices]
                average_direction_array[j] = np.mean(ending_directions, axis=0)
            detector_dir_list[i] = average_direction_array                        
        
        print 'culprit_count: ' + str(culprit_count)

        writer = AngleRootWriter(filename)
        for i in range(self.npmt_bins):
            writer.write_PMT(detector_dir_list[i], i, self.detectorxbins, self.detectorybins, self.detectorzbins)
            #print writer.T.GetEntries()
        writer.close()

        return detector_dir_list

    def _scaled_pmt_arr_surf(self, closest_triangle_index):
        closest_triangle_index = np.asarray(closest_triangle_index)
        curved_surface_index = (closest_triangle_index/self.n_triangles_per_surf).astype(int)
        renorm_triangle = closest_triangle_index % self.n_triangles_per_surf

        mtx = (np.tile(2 * self.c_rings_rolled, (len(renorm_triangle), 1)).T - renorm_triangle).T
        pixels_outside_hit_ring = self.c_rings_rolled[np.argmax(mtx > 0, axis=1) - 1]   # This may not be the right name for this??
        complicated = ((renorm_triangle - 2*pixels_outside_hit_ring) % self.ring[np.argmax(mtx>0,axis=1)-1])

        pmt_number = complicated + pixels_outside_hit_ring + curved_surface_index*self.n_pmts_per_surf

        ### Alternative calculation ###
        new_lens_number = (pmt_number / self.n_pmts_per_surf).astype(int)
        pixel_number_in_lens = pmt_number - curved_surface_index*self.n_pmts_per_surf       # Depends on PMT number...
        ring = np.searchsorted(self.c_rings, pixel_number_in_lens, side='right')
        pixel_number_in_ring = pixel_number_in_lens - self.c_rings_rolled[ring]
        pixels_outside_hit_ring2 = self.c_rings_rolled[ring]
        pmt_number2 = pixel_number_in_ring + pixels_outside_hit_ring2 + curved_surface_index*self.n_pmts_per_surf

        for i in range(len(closest_triangle_index)):
            if new_lens_number[i] != curved_surface_index[i]:
                print('Lens number mismatch: %d, %d, %d' % (i, new_lens_number[i], new_lens_number[i]))
            if pmt_number[i] != pmt_number2[i]:
                print('Pixel number mismatch: %d, %d, %d, %d' % (i, pmt_number[i], pmt_number2[i], pmt_number[i] - pmt_number2[i]))
            if pixel_number_in_ring[i] != complicated[i]:
                print('Pixel number in ring mismatch: %d, %d, %d, %d' % (i, pixel_number_in_ring[i], complicated[i], pixel_number_in_ring[i] - complicated[i]))
            if pixels_outside_hit_ring[i] != pixels_outside_hit_ring2[i]:
                print('Pixel number outside ring mismatch: %d, %d, %d, %d, ring: %d' % (i, pixels_outside_hit_ring[i], pixels_outside_hit_ring2[i], pixels_outside_hit_ring[i] - pixels_outside_hit_ring2[i], ring[i]))

        return pmt_number, curved_surface_index, ring, pixel_number_in_ring

    def find_pmt_bin_array_new(self, pos_array):
        closest_triangle_index, closest_triangle_dist = self.find_closest_triangle_center(pos_array, max_dist=1.)
        pmts, lenses, rings, pixels = self._scaled_pmt_arr_surf(closest_triangle_index)
        bad_bins = np.asarray(np.where(pmts >= self.npmt_bins))  # Why does this have to be an array inside of an array?  How to convert a tuple into an array? asarray() shuld do it
        if np.size(bad_bins) > 0:
            print('Bad bin count: ' + str(len(bad_bins[0])))
            print("The following " + str(np.shape(bad_bins)[1]) + " photons were not associated to a PMT: " + str(bad_bins))
            pmts = np.delete(pmts, bad_bins) # Note: this line wont work with new scaled_pmt_arr_surf scheme, and it also breaks calibration
            lenses = np.delete(lenses, bad_bins)
            rings = np.delete(rings, bad_bins)
            pixels = np.delete(pixels, bad_bins)
            print('New bin array length: ' + str(len(pmts)))
        return pmts, lenses, rings, pixels

    def find_pmt_bin_array(self, pos_array):
        if(self.detector_r == 0):   # Note: this code is appears to be specific to the icosahedron
            # returns an array of global pmt bins corresponding to an array of end-positions
            length = np.shape(pos_array)[0]
            #facebin array is left as -1s, that way if a particular photon does not get placed onto a side, it gets ignored (including its pmt_position) in the checking stages at the bottom of this function.
            facebin_array = -np.ones(length).astype(int)
            pmt_positions = np.empty((length, 3))
            bin_array = np.empty(length)
            transpose_pos_array = np.transpose(pos_array)
            
            # finding which PMT face each position belongs to by seeing which inverse rotation and displacement brings the position closest to z = 0
            for k in range(20):
                initial_position_array = np.transpose(np.dot(self.inverse_rotation_matrices[k], transpose_pos_array)) - self.inverse_rotated_displacement_matrix[k]

                #creating the facebin_array and pmt_positions based upon if for this value of k the initial_position of the zcoord is close to 0.
                wherezeros = (initial_position_array[:,2] < 1e-5) & (initial_position_array[:,2] > -1e-5)
                reshapedwherezeros = np.column_stack((wherezeros, wherezeros, wherezeros))
                np.place(facebin_array, wherezeros, k)
                np.copyto(pmt_positions, initial_position_array, where=reshapedwherezeros)

            xbin_array = np.floor(self.calc1*pmt_positions[:,0] + self.calc2)
            ybin_array = np.floor(self.calc3*pmt_positions[:,1] + self.calc4)
            
            #making a single bin index from the x, y and facebins.
            bin_array = facebin_array*self.calc5 + ybin_array*self.pmtxbins + xbin_array
            #returning -1 for any bad or impossible bins
            for i in range(length):
                if (xbin_array[i] >= self.pmtxbins) or (xbin_array[i] < 0) or (ybin_array[i] >= self.pmtybins) or (ybin_array[i] < 0) or (facebin_array[i] == -1):
                    bin_array[i] = -1
                    print("photon " + str(i) + " is a culprit")
            return bin_array.astype(int)
            
        else:
            #print("Curved surface detector was selected.")
            closest_triangle_index, closest_triangle_dist = self.find_closest_triangle_center(pos_array, max_dist=1.)
            bin_array, _, _, _ = self._scaled_pmt_arr_surf(closest_triangle_index)
            print('Bin array length: ' + str(len(bin_array)))
            #curved_surface_index = [int(x / self.n_triangles_per_surf) for x in closest_triangle_index]
            #surface_pmt_index = [((x % self.n_triangles_per_surf) % (self.n_pmts_per_surf)) for x in closest_triangle_index]
            #bin_array = [((x*self.n_pmts_per_surf) + y) for x,y in zip(curved_surface_index,surface_pmt_index)]
            ba2 = np.asarray(bin_array)
            bad_bins = np.asarray(np.where(ba2 >= self.npmt_bins))  # Why does this have to be an array inside of an array?  How to convert a tuple into an array? asarray() shuld do it
            #print np.array(bin_array) >= n_pmts_total
            #print np.extract((np.array(bin_array) >= n_pmts_total), bin_array)
            if np.size(bad_bins) > 0:
                print('Bad bin count: ' + str(len(bad_bins[0])))
                print("The following "+str(np.shape(bad_bins)[1])+" photons were not associated to a PMT: " + str(bad_bins))
                # Note: Deleting these will break calibration
                #bin_array = np.delete(bin_array, bad_bins[0])
                print('New bin array length: ' + str(len(bin_array)))
                #print max(closest_triangle_index)
                #print max(bin_array)
                #print n_pmts_total
                #print "Distances to nearest PMT: "
                #print closest_triangle_dist[bad_bins[0]] # Determine distances correctly
            #fig = plt.figure(figsize=(15, 10))
            #plt.hist(bin_array,bins=6*20)
            #plt.xlabel("PMT index")
            #plt.ylabel("counts")
            #plt.show()
            #for ii in range(len(bin_array)):
                #print ii, "\t", closest_triangle_index[ii],"\t",curved_surface_index[ii], "\t",surface_pmt_index[ii], "\t",bin_array[ii]    
            print('bin_array: ' + str(bin_array))
            return bin_array.astype(int)
            
    def find_closest_triangle_center(self, pos_array, max_dist = 1.):
        #print "Finding closest triangle centers..."
        if(max_dist == 1.):
            max_dist = 1.1*2*np.pi*self.lns_rad/self.nsteps # Circumference of detecting surface divided by number of steps, with 1.1x of wiggle room
        #max_dist = 1
        #max_dist=1000
        query_results = self.triangle_centers_tree.query(pos_array,distance_upper_bound = max_dist)
        print('Tree query results: ' + str(query_results))
        closest_triangle_index = query_results[1].tolist()
        closest_triangle_dist = query_results[0].tolist()
        # print('Triangle index: ' + str(closest_triangle_index))
        # print('Triangle distance: ' + str(closest_triangle_dist))
        #print max(closest_triangle_dist)
        
        #fig = plt.figure(figsize=(15, 10))
        #plt.hist(closest_triangle_dist,bins=100)
        #plt.xlabel("distance to closest triangle")
        #plt.ylabel("counts")
        #plt.show()
        #quit()
        
        #fig = plt.figure(figsize=(15, 10))
        #ax = fig.add_subplot(111, projection='3d')
        #for i in range(len(pos_array[:,0])):
            #if(closest_triangle_dist[i]<float('inf')):
                ##print i, closest_triangle_index[i],closest_triangle_dist[i]
                #x = [pos_array[i,0],self.triangle_centers[closest_triangle_index[i],0]]
                #y = [pos_array[i,1],self.triangle_centers[closest_triangle_index[i],1]]
                #z = [pos_array[i,2],self.triangle_centers[closest_triangle_index[i],2]]
                #ax.scatter(pos_array[i,0], pos_array[i,1], pos_array[i,2], color='b')
                #ax.scatter(self.triangle_centers[closest_triangle_index[i],0], self.triangle_centers[closest_triangle_index[i],1], self.triangle_centers[closest_triangle_index[i],2], color='r')
                #ax.plot(x,y,z, color='g')
        #ax.scatter(self.triangle_centers[:,0].tolist(), self.triangle_centers[:,1].tolist(), self.triangle_centers[:,2].tolist(),s=5, c='m')
        #ax.set_xlabel('X Label')
        #ax.set_ylabel('Y Label')
        #ax.set_zlabel('Z Label')
        #axis_range = 10
        #ax.set_xlim([-axis_range,axis_range])
        #ax.set_ylim([-axis_range,axis_range])
        #ax.set_zlim([-axis_range,axis_range])
        #plt.show()	
        
        return closest_triangle_index, closest_triangle_dist

    def plot_pdf(self, pdf, plot_title, photon_start=None, photon_end=None, bin_pos=None, show=True):
        # Returns a 3D plot of detector pdf, with each point's size proportional to its
        # probability. Can include list of photon start/end positions to draw in red.
        # Can also include a list of bin positions to draw PDF in real space
        # instead of bin index; should use this if photon_pos is used.
        # If modifying the figure that's returned, use show=False to allow later plotting.
        nonzero_entries = np.nonzero(pdf)
        if bin_pos is None:
            xs = nonzero_entries[0]
            ys = nonzero_entries[1]
            zs = nonzero_entries[2]
        else:
            bin_pos = np.reshape(bin_pos,np.shape(pdf)+(3,))
            xs = bin_pos[nonzero_entries[0],0,0,0]
            ys = bin_pos[0,nonzero_entries[1],0,1]
            zs = bin_pos[0,0,nonzero_entries[2],2]

        sizes = 10000*pdf[nonzero_entries]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs, s=sizes)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(plot_title)

        # Plot lines tracking photon path, with points at the detection position
        if not (photon_start is None or photon_end is None):
            max_ph = 50 # Restrict to only the first max_ph photons, to avoid drawing overload
            if np.shape(photon_end)[0] > max_ph:
                photon_end = photon_end[:max_ph,:]
            if np.shape(photon_start)[0] > max_ph:
                photon_start = photon_start[:max_ph,:]
            ax.scatter(photon_end[:,0],photon_end[:,1],photon_end[:,2],color='red')
            for ii in range(np.shape(photon_end)[0]):
                x_ph = [photon_start[ii,0],photon_end[ii,0]]
                y_ph = [photon_start[ii,1],photon_end[ii,1]]
                z_ph = [photon_start[ii,2],photon_end[ii,2]]
                ax.plot(x_ph,y_ph,z_ph,color='red')
        if show:
            plt.show()
        return fig
    
    @staticmethod
    def detectorbin_index_to_tuple(detectorindex):
        #takes integer values for a detector index (a number from 0 to detxbins*detybins*detzbins - 1) and detnbins and returns a tuple of the xbin, ybin, and zbin.
        xbin = (detectorindex % (self.detectorxbins*self.detectorybins)) % self.detectorxbins
        ybin = ((detectorindex-xbin) % (self.detectorxbins*self.detectorybins))/self.detectorxbins
        zbin = (detectorindex-xbin-ybin*self.detectorxbins)/(self.detectorxbins*self.detectorybins)
        return (xbin, ybin, zbin)
    
    @staticmethod
    def pmt_bin_to_tuple(pmtbin, xbins, ybins):
        #takes a pmtbin number (or array of numbers) and returns the tuple of its facebin, xbin, and ybin.
        xbin = pmtbin % xbins
        ybin = ((pmtbin-xbin)/xbins) % ybins
        facebin = (pmtbin-ybin*xbins-xbin)/(xbins*ybins)
        return (facebin, xbin, ybin)

    def pmt_bin_to_position(self, pmtbin):
        #input a pmtbin number (or array of pmtbin numbers) to output its coordinate center in 3d-space. 
        #init_coord is the initial coordinate of the pmt center before it is rotated and displaced 
        #onto the correct pmtface. 
        if(self.detector_r==0):
			facebin, xbin, ybin = self.pmt_bin_to_tuple(pmtbin, self.pmtxbins, self.pmtybins)
			init_xcoord = self.pmt_side_length/(2.0*self.pmtxbins)*(2*xbin+1) - self.pmt_side_length/2.0
			init_ycoord = np.sqrt(3)*self.pmt_side_length/(4.0*self.pmtybins)*(2*ybin+1) - np.sqrt(3)*self.pmt_side_length/6.0
			init_coord = np.array([init_xcoord, init_ycoord, np.zeros(np.shape(init_xcoord))])
			if len(np.shape(init_coord)) == 1: # Handle a single bin differently, as it has fewer dimensions
				bin_coord = np.dot(self.rotation_matrices[facebin], init_coord) + self.displacement_matrix[facebin]
			else:
				bin_coord = np.einsum('ijk,ki->ij',self.rotation_matrices[facebin], init_coord) + self.displacement_matrix[facebin]
			return bin_coord
        else:
			# Get indices of curved surfaces hit, then translate to first of two triangles hit
			#print self.n_triangles_per_surf
			#print self.n_pmts_per_surf
			curved_surf_bin = (pmtbin/self.n_pmts_per_surf) # Integer division - drops remainder
			#print curved_surf_bin
			#print max(curved_surf_bin)

			triangle_bin1 = curved_surf_bin*self.n_triangles_per_surf + (pmtbin % self.n_pmts_per_surf)
			triangle_bin2 = triangle_bin1+self.n_pmts_per_surf # Second triangle for this PMT
			
			triangle_pos1 = self.triangle_centers[triangle_bin1]
			triangle_pos2 = self.triangle_centers[triangle_bin2]

			#print triangle_bin1
			#print triangle_pos1
			bin_pos = (triangle_pos1+triangle_pos2)/2.0

			return bin_pos
   
    def bin_to_position(self, bintup):
        #takes a detector bin tuple and outputs the coordinate position tuple at the CENTER of the bin
        #origin is at center of whole configuration
        xpos = (bintup[0]+(1.0-self.detectorxbins)/2.0)*(2.0*self.inscribed_radius/self.detectorxbins)
        ypos = (bintup[1]+(1.0-self.detectorybins)/2.0)*(2.0*self.inscribed_radius/self.detectorybins)
        zpos = (bintup[2]+(1.0-self.detectorzbins)/2.0)*(2.0*self.inscribed_radius/self.detectorzbins)
        return (xpos,ypos,zpos)

    def bin_to_position_array(self):
        #returns an array of coordinate centers for each bin.
        x, y, z = np.mgrid[0:self.detectorxbins, 0:self.detectorybins, 0:self.detectorzbins]
        xpos = (x+(1.0-self.detectorxbins)/2.0)*(2.0*self.inscribed_radius/self.detectorxbins)
        ypos = (y+(1.0-self.detectorybins)/2.0)*(2.0*self.inscribed_radius/self.detectorybins)
        zpos = (z+(1.0-self.detectorzbins)/2.0)*(2.0*self.inscribed_radius/self.detectorzbins)
        position_array = zip(xpos.ravel(), ypos.ravel(), zpos.ravel())
        return position_array
                
    def write_to_ROOT(self, filename):
        print "Base class DetectorResponse has nothing to write - instantiate as a subclass."

    def read_from_ROOT(self, filename):
        self.is_calibrated = True
        print "Base class DetectorResponse has nothing to read - instantiate as a subclass."
            
