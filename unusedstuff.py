    def find_pmt_bin(self, pos):
        initial_positions = np.empty((20, 3))
        for k in range(20):
            initial_positions[k] = np.dot(self.inverse_rotation_matrices[k], pos) - self.rotated_displacement[k]
        face_bin = np.argmin(np.absolute(initial_positions[:,2]))

        if initial_positions[face_bin, 2] > 1e-5:
            return -1
        xbin = np.floor((initial_positions[face_bin, 0]+self.pmt_side_length/2)*self.pmtxbins/self.pmt_side_length)
        ybin = np.floor(2*(initial_positions[face_bin, 1]+np.sqrt(3.0)/6*self.pmt_side_length)*self.pmtybins/(np.sqrt(3)*self.pmt_side_length))
        #returning -1 for any impossible bins
        if (xbin >= self.pmtxbins) or (xbin < 0) or (ybin >= self.pmtybins) or (ybin < 0):
            return -1
        #making a single bin index from the x, y and facebins.
        bin = int(face_bin*self.pmtxbins*self.pmtybins + ybin*self.pmtxbins + xbin)
        return bin 

    def find_pmt_bin2(self, pos):
        facebin = 0
        lowest_position = np.dot(self.inverse_rotation_matrices[0], pos) - self.rotated_displacement[0]
        for k in range(1, 20):
            initial_position = np.dot(self.inverse_rotation_matrices[k], pos) - self.rotated_displacement[k]
            if initial_position[2] < lowest_position[2]:
                facebin = k
                lowest_position = initial_position
        if lowest_position[2] > 1e-5:
            return -1
        xbin = np.floor((lowest_position[0]+self.pmt_side_length/2)*self.pmtxbins/self.pmt_side_length)
        ybin = np.floor(2*(lowest_position[1]+np.sqrt(3.0)/6*self.pmt_side_length)*self.pmtybins/(np.sqrt(3)*self.pmt_side_length))
        #returning -1 for any impossible bins
        if (xbin >= self.pmtxbins) or (xbin < 0) or (ybin >= self.pmtybins) or (ybin < 0):
            return -1
        #making a single bin index from the x, y and facebins.
        bin = int(facebin*self.pmtxbins*self.pmtybins + ybin*self.pmtxbins + xbin)
        return bin

    def find_pmt_bin3(self, pos):
        for k in range(20):
            initial_position = np.dot(self.inverse_rotation_matrices[k], pos) - self.rotated_displacement[k]
            if initial_position[2] < 1e-5:
                facebin = k
                break
        xbin = np.floor(self.calc1*initial_position[0] + self.calc2)
        ybin = np.floor(self.calc3*initial_position[1] + self.calc4)
        #returning -1 for any impossible bins
        if (xbin >= self.pmtxbins) or (xbin < 0) or (ybin >= self.pmtybins) or (ybin < 0):
             return -1
        #making a single bin index from the x, y and facebins.
        bin = facebin*self.calc5 + ybin*self.pmtxbins + xbin
        return int(bin)

def find_pmt_bin_array_old(self, pos_array):
        # returns an array of pmt bins corresponding to an array of end-positions
        length = np.shape(pos_array)[0]
        facebin_array = np.empty(length)
        pmt_positions = np.empty((length, 3))
        bin_array = np.empty(length)
        transpose_pos_array = np.transpose(pos_array)
        
        # finding which pmt face each position belongs to by seeing which inverse rotation and displacement brings the position closest to z = 0
        for k in range(20):
            initial_position_array = np.transpose(np.dot(self.inverse_rotation_matrices[k], transpose_pos_array)) - self.rotated_displacement[k]
            for i in range(length):
                initial_zcoord = initial_position_array[i,2]

                if (initial_zcoord < 1e-5) and (initial_zcoord > -1e-5):
                    facebin_array[i] = k
                    pmt_positions[i] = initial_position_array[i]

        xbin_array = np.floor(self.calc1*pmt_positions[:,0] + self.calc2)
        ybin_array = np.floor(self.calc3*pmt_positions[:,1] + self.calc4)
        
        #making a single bin index from the x, y and facebins.
        bin_array = facebin_array*self.calc5 + ybin_array*self.pmtxbins + xbin_array
        #returning -1 for any impossible bins
        for i in range(length):
            if (xbin_array[i] >= self.pmtxbins) or (xbin_array[i] < 0) or (ybin_array[i] >= self.pmtybins) or (ybin_array[i] < 0):
                bin_array[i] = -1
                print "bin " + str(i) + " is a culprit"
        return bin_array.astype(int)



    def build_pdfs(self, detectorxbins, detectorybins, detectorzbins, filename):
        reader = ShortRootReader(filename)
        detectorhits = np.zeros((detectorxbins, detectorybins, detectorzbins))
        calc6x = detectorxbins/(2*self.inscribed_radius)
        calc6y = detectorybins/(2*self.inscribed_radius)
        calc6z = detectorzbins/(2*self.inscribed_radius)
        calc7x = detectorxbins/2.0
        calc7y = detectorybins/2.0
        calc7z = detectorzbins/2.0
        #pmt_initial_pos = [[] for i in range(self.npmt_bins)]
        count = 0
        for ev in reader:
            detected = (ev.photons_end.flags & (0x1 <<2)).astype(bool)
            beginning_photons = ev.photons_beg.pos[detected]
            ending_photons = ev.photons_end.pos[detected]
            
            #finding the bin indices
            for i in range(np.shape(ending_photons)[0]):
                bin_ind = self.find_pmt_bin3(ending_photons[i])
                
                #bin_ind = i%(1138)
                if bin_ind == -1:
                    print "Oh no"
                    continue
                xinitbin = int(np.floor(calc6x*beginning_photons[i,0]+calc7x))
                yinitbin = int(np.floor(calc6y*beginning_photons[i,1]+calc7y))
                zinitbin = int(np.floor(calc6z*beginning_photons[i,2]+calc7z))
                
                #detectorbin = int(zinitbin*detectorxbins*detectorybins + yinitbin*detectorxbins + xinitbin)
                #detectorhits[xinitbin, yinitbin, zinitbin] += 1

                #adding the TH3F objects to the list pmt_inital_pos
                if self.pdfs[bin_ind] == []:
                    #hist3D = TH3F('hist'+str(bin_ind), '', detectorxbins, -inscribed_radius, inscribed_radius, detectorybins, -inscribed_radius, inscribed_radius, detectorzbins, -inscribed_radius, inscribed_radius)
                    #hist3D.Fill(beginning_photons[i][0], beginning_photons[i][1], beginning_photons[i][2])
                    #pmt_initial_pos[bin_ind] = hist3D 
                    #pmt_initial_pos[bin_ind] = np.expand_dims(beginning_photons[i], axis=0)
                    self.pdfs[bin_ind] = detectorhits 
                    self.pdfs[bin_ind][xinitbin, yinitbin, zinitbin] = 1
                    count += 1
                else:
                    #pmt_initial_pos[bin_ind].Fill(beginning_photons[i][0], beginning_photons[i][1], beginning_photons[i][2])
                    #pmt_initial_pos[bin_ind] = np.vstack((pmt_initial_pos[bin_ind], beginning_photons[i]))
                    self.pdfs[bin_ind][xinitbin, yinitbin, zinitbin] += 1
            break 
        print count
    
def create_plots(self, beginning_photons, pmt_bin_array, pmt_choice):
        choicephotons = np.where(pmt_bin_array == pmt_choice)[0]
        choicebeginning = np.empty((np.size(choicephotons), 3))
        for i in range(np.size(choicephotons)):
            choicebeginning[i] = beginning_photons[choicephotons][i]
        if np.size(choicebeginning) == 0:
            print "pmt_bin " + str(pmt_choice) + " is empty"
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            xs = choicebeginning[:,0]
            ys = choicebeginning[:,1]
            zs = choicebeginning[:,2]
            ax.scatter(xs, ys, zs)
            ax.set_xlabel('X initial position')
            ax.set_ylabel('Y initial position')
            ax.set_zlabel('Z initial position')
            plt.title('Initial Position of Photons for Bin ' + str(pmt_choice))
            plt.show()

            fig = plt.figure(figsize=(7.8, 6))
            plt.hist2d(choicebeginning[:,0], choicebeginning[:,1], bins=100)
            plt.xlabel('x pos')
            plt.ylabel('y pos')
            plt.title('pdf')
            plt.colorbar()
            plt.show()   

   # def create_3dplot(self, pmt_choice, nxbins, nybins, nzbins):
    #      selections = np.where(self.totalpmtbins == pmt_choice)[0]
    #      array_length = np.size(selections)
    #      xbins = np.empty(array_length)
    #      ybins = np.empty(array_length)
    #      zbins = np.empty(array_length)
    #      sizes = np.empty(array_length)
    #      for i in range(array_length):
    #          index = selections[i]
    #          xbins[i] = self.totalxbins[index]
    #          ybins[i] = self.totalybins[index]
    #          zbins[i] = self.totalzbins[index]
    #          sizes[i] = 10000*self.pdfs[int(pmt_choice)][xbins[i], ybins[i], zbins[i]]
    #      xs = (2*xbins-1)*self.inscribed_radius/(2.0*nxbins)
    #      ys = (2*ybins-1)*self.inscribed_radius/(2.0*nybins)
    #      zs = (2*zbins-1)*self.inscribed_radius/(2.0*nzbins)
         
    #      fig = plt.figure()
    #      ax = fig.add_subplot(111, projection='3d')
    #      ax.scatter(xs, ys, zs, s=sizes)
    #      ax.set_xlabel('X')
    #      ax.set_ylabel('Y')
    #      ax.set_zlabel('Z')
    #      plt.title('Initial Position of Photons for Bin ' + str(pmt_choice))
    #      plt.show()

  ##to be placed under def __init__ of DetectorResponse in conjuction with create3dplot:

        # self.totalxbins = np.empty(0)
        # self.totalybins = np.empty(0)
        # self.totalzbins = np.empty(0)
        # self.totalpmtbins = np.empty(0)

 ##to be placed inside the buildppdfs function after zinitbin_array is created in conjuction with create3dplot:

            # if np.size(self.totalpmtbins) == 0:
            #     self.totalxbins = xinitbin_array
            #     self.totalybins = yinitbin_array
            #     self.totalzbins = zinitbin_array
            #     self.totalpmtbins = pmt_bin_array
            # else:
            #     self.totalxbins = np.concatenate((self.totalxbins, xinitbin_array))
            #     self.totalybins = np.concatenate((self.totalybins, yinitbin_array))
            #     self.totalzbins = np.concatenate((self.totalzbins, zinitbin_array))
            #     self.totalpmtbins = np.concatenate((self.totalpmtbins, pmt_bin_array))

      #self.create_3dplot(2632, detectorxbins, detectorybins, detectorzbins)

 #detectorbin = (zinitbin*detectorxbins*detectorybins + yinitbin*detectorxbins + xinitbin).astype(int)


  def find_max_starting_bin(self, bin_array, event_pdf, shape):
        #Finds the bin that produces the largest average value among it and its immediate neighbors
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

            value = event_pdf[bintup]
            number_of_bins = 7.0
            for neighbor in range(6):
                if np.sum(neighbors[neighbor]) == -3:
                    number_of_bins += -1
                    continue
                indextup = tuple(neighbors[neighbor])
                value += event_pdf[indextup]

            #taking the average so that bins on the edge are not penalized:
            value = value/number_of_bins

            if value > max_value:
                max_value = value
                max_starting_bin = bintup
        return max_starting_bin



# def find_max_starting_bin(bin_array, event_pdf):
#     '''Finds the bin that when added to its immediate neighbors produces the largest value. Called find_max_starting_bin as opposed to find_max_initial_first_derivative, because technically if the starting bin contains most of the probability relative to its neighbors (while still having the highest inital+neighbors total out of any possible starting bin), then the first derivative might not be the largest, compared to if we started with a neighbor and then encapsulated the starting bin as a secondary point. Although more than likely I would bet that the two would be the same.'''
#     max_starting_bin = (0, 0, 0)
#     max_value = 0
#     for bintup in bin_array:
#         value = 0
#         neighbors = -np.ones((6, 3))
#         for i in range(3):
#             # replace 4 with shape[i]
#             if (bintup[i] != 4-1):
#                 neighbors[2*i] = bintup
#                 neighbors[2*i][i] += 1
#             if (bintup[j] != 0):
#                 neighbors[2*i+1] = bintup
#                 neighbors[2*i+1][i] += -1
#         for neighbor in range(6):
#             if np.sum(neighbors[neighbor]) == -3:
#                 continue
#             value = value + event_array[neighbors[neighbor]]
#         if max_value < value:
#             max_value = value
#             max_starting_bin = bintup
#     return max_starting_bin

# def find_max_starting_zone(bin_array):
#     '''called find_max_starting_zone as opposed to find_max_initial_first_derivative, because technically if the starting bin contains most of the probability relative to its neighbors (while still having the highest inital+neighbors total out of any possible starting bin), then the first derivative might not be the largest, compared to if we started with a neighbor and then encapsulated the starting bin as a secondary point. Although more than likely I would bet that the two would be the same.'''
#     max_starting_bin = (0, 0, 0)
#     max_value = 0
    # for i in bin_array:
    #     if i[0] == shape[0]:
    #         right_bin = -1
    #     else:
    #         right_bin = np.add(i, [1, 0, 0])
    #     if i[0] == 0:
    #         left_bin = -1
    #     else:
    #         left_bin = np.add(i, [-1, 0, 0])
    #     if i[1] == shape[1]:
    #         forward_bin = -1
    #     else:
    #         forward_bin = np.add(i, [0, 1, 0])
    #     if i[1] == 0:
    #         back_bin = -1
    #     else:
    #         back_bin = np.add(i, [0, -1, 0])
    #     if i[2] == shape[2]:
    #         up_bin = -1
    #     else:
    #         up_bin = np.add(i, [0, 0, 1])
    #     if i[2] == 0:
    #         down_bin = -1
    #     else:
    #         down_bin = np.add(i, [0, 0, -1])
    
#     for bintup in bin_array:
#         value = 0
#         newneighbors = -np.ones((6, 3))
#         for i in range(3):
#             # replace 4 with shape[i]
#             if (bintup[i] != 4-1):
#                 newneighbors[2*i] = bintup
#                 newneighbors[2*i][i] += 1
#             if (bintup[j] != 0):
#                 newneighbors[2*i+1] = bintup
#                 newneighbors[2*i+1][i] += -1
#         for neighbor in range(6):
#             if np.sum(newneighbors[neighbor]) == -3:
#                 continue
#             value = value + event_array[newneighbors[neighbor]]
#         if max_value < value:
#             max_value = value
#             max_starting_bin = bintup
#     return max_starting_bin


#old kabamland gaussian smear that is wrong:
def gaussian_sphere2(pos, sigma, n):
    # constructs an initial distribution of photons in a sphere where each photon's radius is chosen by sigma in a guassian.  (choosing each photon's radius uniformly does not create a uniform sphere, so I'm not sure if choosing each photons radius by a normal distribution will be a "normally distributed sphere").
    # math is subject to being incorrect
    radii = np.random.normal(0.0, sigma, n)
    theta = np.arccos(np.random.uniform(-1.0, 1.0, n))
    phi = np.random.uniform(0.0, 2*np.pi, n)
    points = np.empty((n, 3))
    points[:,0] = radii*np.sin(theta)*np.cos(phi) + pos[0]
    points[:,1] = radii*np.sin(theta)*np.sin(phi) + pos[1]
    points[:,2] = radii*np.cos(theta) + pos[2]
    pos = points
    dir = uniform_sphere(n)
    pol = np.cross(dir, uniform_sphere(n))
    #300 nm is roughly the pseudocumene scintillation wavelength
    wavelengths = np.repeat(300.0, n)
    return Photons(pos, dir, pol, wavelengths) 

#spherical lens built in stupid way: (why do xs vary as if they were on a sphere of radius diameter/2.0? they aren't on a sphere- they are on a spherical cap.
def spherical_lens2(R1, R2, diameter, nsteps=1024):
    #constructs a spherical lens with specified radii of curvature
    angles1 = np.linspace(-np.pi/2, 0, nsteps/2, endpoint=False)
    angles2 = np.linspace(0, np.pi/2, nsteps/2)
    x1 = diameter/2.0*np.cos(angles1)
    x2 = diameter/2.0*np.cos(angles2)
    y1 = -(np.sqrt(R1**2-x1**2)-np.sqrt(R1**2-(diameter/2.0)**2))
    y2 = (np.sqrt(R2**2-x2**2)-np.sqrt(R2**2-(diameter/2.0)**2))
    return make.rotate_extrude(np.concatenate((x1,x2)), np.concatenate((y1,y2)), nsteps=64)

def spherical_lens3(R1, R2, diameter, nsteps=1024):
    '''constructs a spherical lens with specified radii of curvature. Choosing R2 > 0 creates a meniscus lens; make sure that abs(R2) > abs(R1) in this case. If meniscus, light should go in the larger cap first for better performance- see MOE 4th ed. p.86
    shift is the length of the hemisphere that is cutout to make the spherical cap 
    R1 goes towards positive y, R2 towards negative y
    concatenation needs to keep it going counterclockwise.'''
    shift1 = np.sqrt(R1**2 - (diameter/2.0)**2)
    shift2 = np.sqrt(R2**2 - (diameter/2.0)**2)
    theta1 = np.arctan(shift1/(diameter/2.0))
    theta2 = np.arctan(shift2/(diameter/2.0))
    angles1 = np.linspace(theta1, np.pi/2, nsteps/2)
    angles2 = np.linspace(np.pi/2, theta2, nsteps/2, endpoint=False)
    x1 = R1*np.cos(angles1)
    x2 = np.sign(R2)*R2*np.cos(angles2)
    y1 = R1*np.sin(angles1) - shift1
    y2 = R2*np.sin(angles2) - np.sign(R2)*shift2
    return make.rotate_extrude(np.concatenate((x2,x1)), np.concatenate((y2,y1)), nsteps=64)


    def build_perfect_resolution_direction_listold(self, detectorxbins, detectorybins, detectorzbins, filename):
        # creates a list of average ending direction per detectorbin per pmtbin. Each pmtbin will have an array of detectorbins with an average ending direction of photons eminating from each one for each one. Simulation must just be the pmticosohedron.  Make sure that the focal length is the same for the simulation as it is in this file at the top.
        reader = ShortRootReader(filename)
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
        
        detector_dir_list = [[] for i in range(self.npmt_bins)]
        ending_dir_list = [[[] for j in range(detectorbins)] for i in range(self.npmt_bins)]
        #for each of the entries in detector_dir_list we want an array of length detectorbins where in each slot of each array we have the average ending direction corresponding to that detector (for photons from that detector going to the pmt cooresponding to the entry index of the whole list).
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
            # creating a single bin for each detectorbin.
            detectorbin_array = xinitbin_array + yinitbin_array*detectorxbins + zinitbin_array*detectorxbins*detectorybins
            end_direction_array = normalize(ending_photons-beginning_photons)

            #ending_directions_array = -np.ones((self.npmt_bins, 3))
            for i in range(self.npmt_bins):
                pmt_indices = np.where(pmt_bin_array == i)[0]
                if np.shape(pmt_indices)[0] == 0:
                    culprit_count += 1
                    continue
                detectorbins_for_pmt = detectorbin_array[pmt_indices]
                average_direction_array = -5*np.ones((detectorbins, 3))
                
                for j in range(detectorbins):
                    detector_indices = np.where(detectorbins_for_pmt == j)[0]
                    if np.shape(detector_indices)[0] == 0:
                        continue
                    direction_indices = pmt_indices[detector_indices]
                    ending_directions = end_direction_array[direction_indices]
                    # if ending_dir_list[i] == []:
                    #     ending_dir_list[i] = ending_directions
                    # else:
                    #     np.append(ending_dir_list[i], ending_directions, axis=0)
                    average_direction_array[j] = np.mean(ending_directions, axis=0)
                detector_dir_list[i] = average_direction_array                        

            loops += 1
            print loops
            # if loops == 50:
            #     break
        
        print 'culprit_count: ' + str(culprit_count)
        print detector_dir_list
        return detector_dir_list
