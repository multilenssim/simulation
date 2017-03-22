from chroma.transform import make_rotation_matrix, normalize
from kabamland import find_inscribed_radius, find_focal_length, return_values
from ShortIO.root_short import PDFRootWriter, PDFRootReader, ShortRootReader, AngleRootReader, AngleRootWriter
import detectorconfig
import numpy as np
from scipy import integrate, optimize, ndimage, spatial
import lensmaterials as lm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
##new
from kabamland import find_max_radius

class DetectorResponse(object):
    def __init__(self, configname):
        config = detectorconfig.configdict[configname]
        self.edge_length, self.facecoords, self.direction, self.axis, self.angle, self.spin_angle = return_values(config.edge_length, config.base)
        self.pmtxbins = config.pmtxbins
        self.pmtybins = config.pmtybins
        self.npmt_bins = 20*self.pmtxbins*self.pmtybins
        self.diameter_ratio = config.diameter_ratio
        self.thickness_ratio = config.thickness_ratio
        ##changed
        self.pcdiameter = 2*config.diameter_ratio*find_max_radius(config.edge_length, config.base)
        self.focal_length = 1.00
        #self.focal_length = find_focal_length(config.edge_length, config.base, config.diameter_ratio, config.thickness_ratio)
        ##end changed
        self.pmt_side_length = np.sqrt(3)*(3-np.sqrt(5))*self.focal_length + self.edge_length
        self.inscribed_radius = find_inscribed_radius(self.edge_length)
        self.inverse_rotation_matrices = self.build_inverse_rotation_matrices()
        self.displacement_matrix = self.build_displacement_matrix()
        self.rotated_displacement = self.build_rotated_displacement_matrix()
        self.lens_rotated_displacement = self.build_lensplane_rotated_displacement_matrix()
        self.pdfs = [[] for i in range(self.npmt_bins)]
        self.calc1 = self.pmtxbins/self.pmt_side_length
        self.calc2 = self.pmtxbins/2.0
        self.calc3 = 2*self.pmtybins/(np.sqrt(3)*self.pmt_side_length)
        self.calc4 = self.pmtybins/3.0
        self.calc5 = self.pmtxbins*self.pmtybins
        
    def build_rotation_matrices(self):
        rotation_matrices = np.empty((20, 3, 3))
        for k in range(20):
            rotation_matrices[k] = np.dot(make_rotation_matrix(self.spin_angle[k], self.direction[k]), make_rotation_matrix(self.angle[k], self.axis[k]))
        return rotation_matrices
        
    def build_inverse_rotation_matrices(self):
        rotation_matrices = np.empty((20, 3, 3))
        for k in range(20):
            rotation_matrices[k] = np.dot(make_rotation_matrix(self.spin_angle[k], self.direction[k]), make_rotation_matrix(self.angle[k], self.axis[k]))
        return np.linalg.inv(rotation_matrices)

    def build_displacement_matrix(self):
        displacement_matrix = np.empty((20, 3))
        for k in range(20):
            displacement_matrix[k] = self.facecoords[k] + self.focal_length*normalize(self.facecoords[k])
        return displacement_matrix

    def build_rotated_displacement_matrix(self):
        rotated_displacement = np.empty((20, 3))
        for k in range(20):
            rotated_displacement[k] = np.dot(self.inverse_rotation_matrices[k], self.displacement_matrix[k])
        return rotated_displacement

    def build_lensplane_rotated_displacement_matrix(self):
        lens_rotated_displacement = np.empty((20, 3))
        for k in range(20):
            lens_rotated_displacement[k] = np.dot(self.inverse_rotation_matrices[k], self.facecoords[k])
        return lens_rotated_displacement

    def angles_response(self, config, simname, filename):
        #takes a simulation file and creates an array of angles that photons hit the pmts at. (replace lenses with disk pmts for the simulation to see what angles light hits the lenses at. Light needs to land on an icosahedron face plane so use disks instead of just changing the surface of the lens to detecting.) ev.photons_end.dir[detected] is always 0s as of now because we aren't saving directions in event or simulation files. 
        reader = ShortRootReader(simname)
        total_angles = np.zeros(0)
        loops = 0
        for ev in reader:
            detected = (ev.photons_end.flags & (0x1 <<2)).astype(bool)
            beginning_photons = ev.photons_beg.pos[detected]
            ending_photons = ev.photons_end.pos[detected]
            transpose_pos_array = np.transpose(ending_photons)
            length = np.shape(ending_photons)[0]

            facebin_array = -np.ones(length).astype(int)
            # finding which LENS face each position belongs to by seeing which inverse rotation and displacement brings the position closest to z = 0. Note this is different than finding the PMT face as in find_pmt_bin_array.
            for k in range(20):
                initial_position_array = np.transpose(np.dot(self.inverse_rotation_matrices[k], transpose_pos_array)) - self.lens_rotated_displacement[k]
                wherezeros = (initial_position_array[:,2] < 1e-5) & (initial_position_array[:,2] > -1e-5)
                np.place(facebin_array, wherezeros, k)

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

    def build_perfect_resolution_direction_list(self, detectorxbins, detectorybins, detectorzbins, simname, filename):
        # creates a list of average ending direction per detectorbin per pmtbin. Each pmtbin will have an array of detectorbins with an average ending direction of photons eminating from each one for each one. Simulation must just be the pmticosohedron.  Make sure that the focal length is the same for the simulation as it is in this file at the top.
        reader = ShortRootReader(simname)
        detectorbins = detectorxbins*detectorybins*detectorzbins
        calc6x = detectorxbins/(2*self.inscribed_radius)
        calc6y = detectorybins/(2*self.inscribed_radius)
        calc6z = detectorzbins/(2*self.inscribed_radius)
        calc7x = detectorxbins/2.0
        calc7y = detectorybins/2.0
        calc7z = detectorzbins/2.0
        #count = 0
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
            detector_bin_array = xinitbin_array + yinitbin_array*detectorxbins + zinitbin_array*detectorxbins*detectorybins

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
        #for each of the entries in detector_dir_list we want an array of length detectorbins where in each slot of each array we have the average ending direction corresponding to that detector (for photons from that detector going to the pmt cooresponding to the entry index of the whole list).     
   
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
            writer.write_event(detector_dir_list[i], i, detectorxbins, detectorybins, detectorzbins)
            #print writer.T.GetEntries()
        writer.close()

        return detector_dir_list

    def analyse_perfect_res_event(self, detectorxbins, detectorybins, detectorzbins, eventfile, detector_dir_list_file, event_pos):
        #Takes an event recorded without lenses and makes a list of ending directions for each pmt. A query is made to match this direction with the closest direction to it in the corresponding perfect_res_dir_list. This matches each photon hit for each pmt to a detector that it is believed the photon came from. final_pdf then adds hits to these detector bins (for each pmt) in order to construct the final pdf.

        detector_dir_list = [[] for i in range(self.npmt_bins)]
        first_reader = AngleRootReader(detector_dir_list_file)
        for bin_ind, angle in first_reader:
            detector_dir_list[bin_ind] = angle
        
        reader = ShortRootReader(eventfile)
        for ev in reader:
            detected = (ev.photons_end.flags & (0x1 <<2)).astype(bool) 
            beginning_photons = ev.photons_beg.pos[detected]
            ending_photons = ev.photons_end.pos[detected]
            end_direction_array = normalize(ending_photons-beginning_photons)
            length = np.shape(ending_photons)[0]
            print "number of photons recording hits in event: " + str(length)

            #creating an array of pmt bins for the ending_photons
            event_pmt_bin_array = self.find_pmt_bin_array(ending_photons)

            final_pdf = np.zeros((detectorxbins, detectorybins, detectorzbins))
            nbad_pmt_bins = 0
            for i in range(length):
                if i % 10000 == 0:
                    print str(i) + ' out of ' + str(length)
                event_pmt_bin = event_pmt_bin_array[i]
                end_direction = end_direction_array[i]
                if np.shape(detector_dir_list[event_pmt_bin]) != (1000, 3):
                    nbad_pmt_bins += 1
                    continue
                detector_array = detector_dir_list[event_pmt_bin]
                tree = spatial.KDTree(detector_array)
                nearest_detector_index = tree.query(end_direction)[1]
                if detector_array[nearest_detector_index][0] < -1:
                    print "bad angle"
                    continue
                xbin = (nearest_detector_index % (detectorxbins*detectorybins)) % detectorxbins
                ybin = ((nearest_detector_index-xbin) % (detectorxbins*detectorybins))/detectorxbins
                zbin = (nearest_detector_index-xbin-ybin*detectorxbins)/(detectorxbins*detectorybins)

                final_pdf[xbin, ybin, zbin] += 1
            
            print nbad_pmt_bins    
            final_pdf = np.float32(final_pdf/float(length))
            
            self.plot_pdf(final_pdf, 'Perfect Resolution PDF of ' + str(eventfile))
            self.det_performance(final_pdf, 200)
            self.det_performance(final_pdf, 200, startingtype='mode')
            self.det_performance(final_pdf, 200, startingtype='center_of_mass')
            self.det_performance(final_pdf, 200, startingtype='actual_position', event_location=event_pos)
        return final_pdf
     
    def get_angles(self, r0, r, L):
        # Returns an array of angles between the vectors from r0 (one 3D position) 
        # to r (many 3D positions) and the line pointing along direction L (one 3D vector)
        
        # Get norms of r-r0; get projections (r-r0).L; take arccos(norm/proj)
        #return 
    
    def find_pmt_bin_array(self, pos_array):
        # returns an array of pmt bins corresponding to an array of end-positions
        length = np.shape(pos_array)[0]
        #facebin array is left as -1s, that way if a particular photon does not get placed onto a side, it gets ignored (including its pmt_position) in the checking stages at the bottom of this function.
        facebin_array = -np.ones(length).astype(int)
        pmt_positions = np.empty((length, 3))
        bin_array = np.empty(length)
        transpose_pos_array = np.transpose(pos_array)
        
        # finding which pmt face each position belongs to by seeing which inverse rotation and displacement brings the position closest to z = 0
        for k in range(20):
            initial_position_array = np.transpose(np.dot(self.inverse_rotation_matrices[k], transpose_pos_array)) - self.rotated_displacement[k]

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
                print "bin " + str(i) + " is a culprit"
        return bin_array.astype(int)

    def build_pdfs(self, detectorxbins, detectorybins, detectorzbins, filename):
        # creates a list of a pdf for each pmt describing the likelihood of photons from particular beginning bins to record hits at that pmt
        reader = ShortRootReader(filename)
        #detectorhits = np.zeros((detectorxbins, detectorybins, detectorzbins))
        calc6x = detectorxbins/(2*self.inscribed_radius)
        calc6y = detectorybins/(2*self.inscribed_radius)
        calc6z = detectorzbins/(2*self.inscribed_radius)
        calc7x = detectorxbins/2.0
        calc7y = detectorybins/2.0
        calc7z = detectorzbins/2.0
        count = 0
        culprit_count = 0
        loops = 0
        
        for ev in reader:
            detected = (ev.photons_end.flags & (0x1 <<2)).astype(bool)
            beginning_photons = ev.photons_beg.pos[detected]
            ending_photons = ev.photons_end.pos[detected]
            length = np.shape(ending_photons)[0]
            
            #creating an array of pmt bins for the ending_photons
            pmt_bin_array = self.find_pmt_bin_array(ending_photons)

            # creating arrays for the detector bins of each photon.
            xinitbin_array = np.floor(calc6x*beginning_photons[:,0]+calc7x).astype(int)
            yinitbin_array = np.floor(calc6y*beginning_photons[:,1]+calc7y).astype(int)
            zinitbin_array = np.floor(calc6z*beginning_photons[:,2]+calc7z).astype(int)
                  
            # adding counts of the photons to their respective positions in the pdf. The pdfs are normalized afterwards
            for i in range(length):
                pmt_bin = pmt_bin_array[i]
                if pmt_bin == -1:
                    culprit_count += 1
                    continue
                elif self.pdfs[pmt_bin] == []:
                    self.pdfs[pmt_bin] = np.zeros((detectorxbins, detectorybins, detectorzbins))
                    self.pdfs[pmt_bin][xinitbin_array[i], yinitbin_array[i], zinitbin_array[i]] = 1
                    count += 1
                else:
                    self.pdfs[pmt_bin][xinitbin_array[i], yinitbin_array[i], zinitbin_array[i]] += 1
            loops += 1
            print loops
            # if loops == 50:
            #     break
        
        self.normalize_pdfs()
        print "count " + str(count)
        print "there are " + str(culprit_count) + " culprits"
                
    def normalize_pdfs(self):
        for i in range(self.npmt_bins):
            if self.pdfs[i] != []:
                self.pdfs[i] = np.float32(self.pdfs[i]/float(np.sum(self.pdfs[i])))

    def plot_pdf(self, pdf, plot_title):
        nonzero_entries = np.nonzero(pdf)
        xs = nonzero_entries[0]
        ys = nonzero_entries[1]
        zs = nonzero_entries[2]
        sizes = 10000*pdf[xs,ys,zs]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs, s=sizes)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(plot_title)
        plt.show()

    def analyze_event(self, detectorxbins, detectorybins, detectorzbins, eventfile, event_pos):
        #takes an event and constructs the pdf based upon which pmts were hit.
        reader = ShortRootReader(eventfile)
        for ev in reader:
            detected = (ev.photons_end.flags & (0x1 <<2)).astype(bool) 
            ending_photons = ev.photons_end.pos[detected]
            event_pmt_bin_array = self.find_pmt_bin_array(ending_photons)
            length = np.shape(ending_photons)[0]
            print "number of photons recording hits in event: " + str(length)

            final_pdf = np.zeros((detectorxbins, detectorybins, detectorzbins))
            for i in range(length):
                event_pmt_bin = event_pmt_bin_array[i]
                if (event_pmt_bin == -1) or (self.pdfs[event_pmt_bin] == []):
                    continue
                else:
                    final_pdf = np.add(final_pdf, self.pdfs[event_pmt_bin])
            final_pdf = np.float32(final_pdf/float(length))
            
            # print "number of photons recording hits in event: " + str(length)
            # final_pdf.tofile('finalpdf-for-config-4-at-(6,6,6).txt')
            
            self.plot_pdf(final_pdf, '(adding model) PDF of ' + str(eventfile))
            self.det_performance(final_pdf, 200)
            self.det_performance(final_pdf, 200, startingtype='mode')
            self.det_performance(final_pdf, 200, startingtype='center_of_mass')
            self.det_performance(final_pdf, 200, startingtype='actual_position', event_location=event_pos)
        return final_pdf

    def det_performance(self, event_pdf, n, startingtype='max_starting_zone', event_location=None):
        #takes the pdf for a single event and creates an array and plot showing what portion of the pdf (probability of event) lies within a sphere of radius r centered at the chosen starting position for n evenly spaced values of r.
        shape = np.shape(event_pdf)
        x, y, z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        bin_array = zip(x.ravel(), y.ravel(), z.ravel())

        if startingtype == 'max_starting_zone':
            startingbin = self.find_max_starting_bin(bin_array, event_pdf, shape)
            startingcoord = self.bin_to_position(startingbin, shape[0], shape[1], shape[2])
        elif startingtype == 'mode':
            startingbin = np.unravel_index(np.argmax(event_pdf), (shape[0], shape[1], shape[2]))
            startingcoord = self.bin_to_position(startingbin, shape[0], shape[1], shape[2])
        elif startingtype == 'center_of_mass':
            startingbin = ndimage.measurements.center_of_mass(event_pdf)
            startingcoord = self.bin_to_position(startingbin, shape[0], shape[1], shape[2])
        elif startingtype == 'actual_position':
            startingcoord = event_location
        else:
            print "You need a starting type!"
            
        coord_array = self.bin_to_position_array(shape[0], shape[1], shape[2]) 
        tree = spatial.KDTree(coord_array)

        probabilities = np.zeros(n)
        radii = np.linspace(0, 2*self.inscribed_radius, n)
        radii[0] = 1e-5
        for i in range(n):
            radius = radii[i]
            neighbors = tree.query_ball_point(startingcoord, radius, eps=1e-5)
            for j in range(len(neighbors)):
                bin_tuple = bin_array[neighbors[j]]
                probabilities[i] = probabilities[i] + event_pdf[bin_tuple]
        performance_array = np.reshape(np.dstack((radii, probabilities)), (n, 2))
        print performance_array
           
        def gaussint(radius, sigma):
            return np.sqrt(2/np.pi)/sigma**3*integrate.quad(lambda x: x**2*np.exp(-x**2/(2.0*sigma**2)), 0, radius)[0]
        # popt, pcov = optimize.curve_fit(gaussint, radii, probabilities, p0=3)
        # print popt
        # fit_probabilities = [gaussint(r, popt[0]) for r in radii]
        
        #combine plots and annotate in sigma = popt[0] and radius at 19.87 percent (1 sigma) and 47.78 percent (1.5 sigma)
        #also want to annotate reduced_square_error = np.sum((fit_probabilites-probabilities)**2)/n
        fig = plt.figure(figsize=(8, 8))
        plt.scatter(radii, probabilities)
        plt.xlabel('Radius from ' + startingtype + ': ' + str(startingcoord))
        plt.ylabel('Probability')
        plt.title('Detector Performance')
        plt.show()

        # fig = plt.figure(figsize=(8, 8))
        # plt.scatter(radii, fit_probabilities)
        # plt.xlabel('Radius from ' + startingtype + ': ' + str(startingcoord))
        # plt.ylabel('Probability')
        # plt.title('optim Performance')
        # plt.show()
        return(performance_array)
    
    def gaussint(self, radii, sigma):
        return [np.sqrt(2/np.pi)/sigma**3*integrate.quad(lambda x: x**2*np.exp(-x**2/(2.0*sigma**2)), 0, radius)[0] for radius in radii]
        
    def pmtbin_to_tuple(pmtbin, xbins, ybins):
        #takes a pmtbin number and returns the tuple of its facebin, xbin, and ybin.
        xbin = pmtbin % xbins
        ybin = ((pmtbin-xbin)/xbins) % ybins
        facebin = (pmtbin-ybin*xbins-xbin)/(xbins*ybins)
        return (facebin, xbin, ybin)

    def pmt_bin_to_position(self, pmtbin):
        facebin, xbin, ybin = pmtbin_to_tuple(pmtbin, self.pmtxbins, self.pmtybins)
        
        
        
   
    def bin_to_position(self, bintup, detectorxbins, detectorybins, detectorzbins):
        #takes a detector bin tuple and outputs the coordinate position tuple at the CENTER of the bin
        #origin is at center of whole configuration
        xpos = (bintup[0]+(1.0-detectorxbins)/2.0)*(2.0*self.inscribed_radius/detectorxbins)
        ypos = (bintup[1]+(1.0-detectorybins)/2.0)*(2.0*self.inscribed_radius/detectorybins)
        zpos = (bintup[2]+(1.0-detectorzbins)/2.0)*(2.0*self.inscribed_radius/detectorzbins)
        return (xpos,ypos,zpos)

    def bin_to_position_array(self, detectorxbins, detectorybins, detectorzbins):
        #takes a shape for the detector bins and returns an array of coordinate centers for each bin.
        x, y, z = np.mgrid[0:detectorxbins, 0:detectorybins, 0:detectorzbins]
        xpos = (x+(1.0-detectorxbins)/2.0)*(2.0*self.inscribed_radius/detectorxbins)
        ypos = (y+(1.0-detectorybins)/2.0)*(2.0*self.inscribed_radius/detectorybins)
        zpos = (z+(1.0-detectorzbins)/2.0)*(2.0*self.inscribed_radius/detectorzbins)
        position_array = zip(xpos.ravel(), ypos.ravel(), zpos.ravel())
        return position_array

    def find_max_starting_bin(self, bin_array, event_pdf, shape):
        #Finds the bin that produces the largest average value among it and its immediate neighbors. Returns a tuple for the center of mass of this bin and its immediate neighbors.
        max_starting_bin = (0, 0, 0)
        max_value = 0
        for bintup in bin_array:
            #constructing the neighbors of bintup such that if bintup is on an edge the neighbor "off" the edge isn't counted (it's made into -1s which are continued past later). Another way to do this would be to append any new neighbor that satisfies the property that it isn't off the edge into an array of neighbors. This works too though.
            neighbors = -np.ones((6, 3))
            for i in range(3):
                if (bintup[i] != shape[i]-1):
                    #(shape[i]-1) is the largest possible bin in the ith dimension
                    neighbors[2*i] = bintup
                    neighbors[2*i][i] += 1
                if (bintup[i] != 0):
                    neighbors[2*i+1] = bintup
                    neighbors[2*i+1][i] += -1
            
            number_of_bins = 7
            value = float(event_pdf[bintup])
            center_of_mass = np.array(bintup)*value
            for neighbor in range(6):
                if np.sum(neighbors[neighbor]) == -3:
                    number_of_bins += -1
                    continue
                indextup = tuple(neighbors[neighbor])
                center_of_mass += neighbors[neighbor]*event_pdf[indextup]
                value += event_pdf[indextup]
            if value == 0:
                continue
            center_of_mass = center_of_mass/value
            #taking the average of value so that bins on the edge are not penalized:
            average_value = value/number_of_bins

            if average_value > max_value:
                max_value = average_value
                max_starting_bin = center_of_mass
        return tuple(max_starting_bin)
                
    def write_to_ROOT(self, filename):
        writer = PDFRootWriter(filename)
        for i in range(self.npmt_bins):
            writer.write_event(self.pdfs[i], i)
        writer.close()

    def read_from_ROOT(self, filename):
        reader = PDFRootReader(filename)
        for bin_ind, pdf in reader:
            self.pdfs[bin_ind] = pdf
            
