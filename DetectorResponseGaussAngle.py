from ShortIO.root_short import GaussAngleRootWriter, GaussAngleRootReader, ShortRootReader
from chroma.transform import normalize
import numpy as np
import matplotlib.pyplot as plt

from DetectorResponse import DetectorResponse

class DetectorResponseGaussAngle(DetectorResponse):
    '''Detector calibration information is stored in Gaussian cones for each PMT: 
    the cones are represented as mean angles in 3D space and their uncertainties 
    (sigma for a cone of unit length) for the light hitting that PMT.
    '''
    def __init__(self, configname, detectorxbins=10, detectorybins=10, detectorzbins=10, infile=None):        
        # If passed infile, will automatically read in the calibrated detector mean angles/sigmas
        DetectorResponse.__init__(self, configname, detectorxbins, detectorybins, detectorzbins)
        self.means = np.zeros((3,self.npmt_bins)) 
        self.sigmas = np.zeros(self.npmt_bins)
        if infile is not None:
            self.read_from_ROOT(infile)
                
    def calibrate(self, simname, nevents=-1):
        # Use with a simulation file 'simname' to calibrate the detector
        # Creates a list of mean angles and their uncertainties (sigma for
        # a cone of unit length), one for each PMT
        # There are 1000 events in a typical simulation.
        # Uses all photons hitting a given PMT at once (better estimate of sigma,
        # but may run out of memory in some cases).
        # Will not calibrate PMTs with <n_min hits
         self.is_calibrated = True
         reader = ShortRootReader(simname)

         culprit_count = 0
         loops = 0         
         full_length = 0
         n_min = 10 # Do not calibrate a PMT if <n_min photons hit it
         if nevents < 1:
             nevents = len(reader)
         total_means = np.zeros((self.npmt_bins, 3))
         total_variances = np.zeros((self.npmt_bins))
         amount_of_hits = np.zeros((self.npmt_bins))

         # Loop through events, store for each photon the index of the PMT it hit (pmt_bins)
         # and the direction pointing back to its origin (end_direction_array)
         loops = 0
         end_direction_array = None
         pmt_bins = None
         max_storage = min(nevents*1000000,120000000) #600M is too much, 400M is OK (for np.float32; using 300M)
         end_direction_array = np.empty((max_storage,3),dtype=np.float32) 
         pmt_bins = np.empty(max_storage,dtype=np.float32) 
         n_det = 0
         for ev in reader:

             loops += 1
             if loops > nevents:
                 break
             
             print "Event " + str(loops) + " of " + str(nevents)

             detected = (ev.photons_end.flags & (0x1 <<2)).astype(bool)
             reflected_diffuse = (ev.photons_end.flags & (0x1 <<5)).astype(bool)
             reflected_specular = (ev.photons_end.flags & (0x1 <<6)).astype(bool)
             print "Total detected: " + str(sum(detected*1))
             print "Total reflected: " + str(sum(reflected_diffuse*1)+sum(reflected_specular*1))
             good_photons = detected & np.logical_not(reflected_diffuse) & np.logical_not(reflected_specular)
             print "Total detected and not reflected: " + str(sum(good_photons*1))
             
             beginning_photons = ev.photons_beg.pos[good_photons]
             ending_photons = ev.photons_end.pos[good_photons]
             length = np.shape(ending_photons)[0]
             end_dir = normalize(ending_photons-beginning_photons)
             # if end_direction_array is None:
             #     end_direction_array = end_dir
             # else:
             #     end_direction_array = np.vstack((end_direction_array, end_dir))
             #end_direction_array.append(end_dir)
             if n_det+length > max_storage:
                 print "Too many photons to store in memory; not reading any further events."
                 break
             end_direction_array[n_det:(n_det+length),:] = end_dir
             pmt_b = self.find_pmt_bin_array(ending_photons)
             # if pmt_bins is None:
             #     pmt_bins = pmt_b
             # else:
             #     pmt_bins = np.hstack((pmt_bins, pmt_b))
             #pmt_bins.append(pmt_b)
             pmt_bins[n_det:(n_det+length)] = pmt_b
             n_det += length
             #print 'Photons detected so far: ' + str(n_det)
             #print 'Sample pmt bins: ' + str(pmt_bins[n_det:(n_det+length)])
         end_direction_array.resize((n_det,3))
         pmt_bins.resize(n_det)

         draw_pmt_ind = -1
         #looping through each event in the simulation, in order to save a mean_angle and a variance for each pmt.
         for i in range(self.npmt_bins):
            if i % 10000 == 0:
                print str(i) + ' out of ' + str(self.npmt_bins) + ' PMTs'
            

            pmt_indices = np.where(pmt_bins == i)[0]
            if np.shape(pmt_indices)[0] == 0:
                continue
            angles_for_pmt = end_direction_array[pmt_indices]

            n_angles = np.shape(angles_for_pmt)[0]
            #skipping pmts with <2 photon hits (in which case the variance will be undefined with ddof=1)
            #also skipping if <n_min photon hits
            if n_angles < 2 or n_angles<n_min:
                continue
            norms = np.repeat(1.0, n_angles)
            mean_angle = normalize(np.mean(angles_for_pmt, axis=0))
            
			# For each PMT, get a pair of axes which form an 
            # orthonormal coordinate system with the PMT mean direction
            u_dir = np.cross(mean_angle,np.array([0,0,1]))
            if not (np.dot(u_dir, u_dir) > 0): # In case mean_angle = [0,0,1]
				u_dir = np.cross(mean_angle,np.array([0,1,0]))
            u_dir = normalize(u_dir)
            v_dir = np.cross(mean_angle, u_dir)
            
            u_proj = np.dot(angles_for_pmt, u_dir)
            u_var = np.var(u_proj, ddof=1)
            v_proj = np.dot(angles_for_pmt, v_dir)
            v_var = np.var(v_proj, ddof=1)
            variance = (u_var+v_var)/2.
            
            # Old method, which calculated variance of projected norma, even though
            # the mean of the projections wasn't 0 due to the solid angle factor
            projection_norms = np.dot(angles_for_pmt, mean_angle)
            orthogonal_complements = np.sqrt(np.maximum(norms**2 - projection_norms**2, 0.))
            #variance = np.var(orthogonal_complements, ddof=1)
            
            try:
				draw_pmt_ind = int(draw_pmt_ind)
				if i == draw_pmt_ind or draw_pmt_ind<0:
					# Temporary, to visualize histogram of angles, distances
					#angles = np.arccos(projection_norms)
					#ang_variance = np.var(angles, ddof=1)
					#fig1 = plt.figure(figsize=(7.8, 6))
					#plt.hist(angles, bins=20)
					#plt.xlabel('Angular Separation to Mean Angle')
					#plt.ylabel('Counts per bin')
					#plt.title('Angles Histogram for PMT ' + str(i))
					##plt.show()
					#
					#fig2 = plt.figure(figsize=(7.8, 6))
					#plt.hist(angles, bins=20, weights=1./np.sin(angles))
					#plt.xlabel('Angular Separation to Mean Angle')
					#plt.ylabel('Counts per solid angle')
					#plt.title('Angles Histogram for PMT ' + str(i))
					
					fig3 = plt.figure(figsize=(7.8, 6))
					plt.hist(orthogonal_complements, bins=20)
					plt.xlabel('Normalized Distance to Mean Angle')
					plt.ylabel('Counts per bin')
					plt.title('Distances Histogram for PMT ' + str(i))					
					
					fig4 = plt.figure(figsize=(7.8, 6))
					plt.hist(u_proj, bins=20)
					plt.xlabel('U Distance to Mean Angle')
					plt.ylabel('Counts per bin')
					plt.title('U Distances Histogram for PMT ' + str(i))
					
					fig5 = plt.figure(figsize=(7.8, 6))
					plt.hist(v_proj, bins=20)
					plt.xlabel('V Distance to Mean Angle')
					plt.ylabel('Counts per bin')
					plt.title('V Distances Histogram for PMT ' + str(i))
					plt.show()
					
					#print "Average projected variance: ", variance
					#print "Variance of projected 2D norms: ", np.var(orthogonal_complements, ddof=1)
					draw_pmt_ind = raw_input("Enter index of next PMT to draw; will stop drawing if not a valid PMT index.\n")
            except ValueError:
				pass
            
            total_means[i] = mean_angle
            total_variances[i] = variance
            amount_of_hits[i] = n_angles
            if np.isnan(variance):
                print "Nan for PMT " + str(i)
                # nan_ind = np.where(np.isnan(orthogonal_complements))
                # print nan_ind
                # print projection_norms
                # print orthogonal_complements
                # print angles_for_pmt
                # print variance
                # print mean_angle

         # temporary, for debugging:
         n_hits = np.sum(amount_of_hits, axis=0)
         print "Total hits for calibrated PMTs: " + str(n_hits)
         print "PMTs w/ < n_events hits: " + str(len(np.where(amount_of_hits < nevents)[0])*1.0/self.npmt_bins)
         print "PMTs w/ < n_min hits: " + str(len(np.where(amount_of_hits < n_min)[0])*1.0/self.npmt_bins)
         print "PMTs w/ < 100 hits: " + str(len(np.where(amount_of_hits < 100)[0])*1.0/self.npmt_bins)
         
         # Store final calibrated values
         self.means = -total_means.astype(np.float32).T
         self.sigmas = np.sqrt(total_variances.astype(np.float32))

    def calibrate_old(self, simname, nevents=-1):
        # Use with a simulation file 'simname' to calibrate the detector
        # Creates a list of mean angles and their uncertainties (sigma for
        # a cone of unit length), one for each PMT
        # There are 1000 events in a typical simulation.
        # Older method - estimates variance for each PMT one event at a time
         self.is_calibrated = True
         reader = ShortRootReader(simname)

         culprit_count = 0
         loops = 0         
         full_length = 0
         if nevents < 1:
             nevents = len(reader)
         total_means = np.zeros((nevents, self.npmt_bins, 3),dtype=np.float32)
         total_variances = np.zeros((nevents, self.npmt_bins),dtype=np.float32)
         amount_of_hits = np.zeros((nevents, self.npmt_bins),dtype=np.float32)

         #looping through each event in the simulation, in order to save a mean_angle and a variance for each pmt.
         for ev in reader:
            print "Event " + str(loops+1) + " of " + str(nevents)
            detected = (ev.photons_end.flags & (0x1 <<2)).astype(bool)
            beginning_photons = ev.photons_beg.pos[detected]
            ending_photons = ev.photons_end.pos[detected]
            length = np.shape(ending_photons)[0]
            end_direction_array = normalize(ending_photons-beginning_photons)
            pmt_bins = self.find_pmt_bin_array(ending_photons)
            
            for i in range(self.npmt_bins):
                if i % 10000 == 0:
                    print str(i) + ' out of ' + str(self.npmt_bins) + ' PMTs'
                pmt_indices = np.where(pmt_bins == i)[0]
                if np.shape(pmt_indices)[0] == 0:
                    continue
                angles_for_pmt = end_direction_array[pmt_indices]
                n_angles = np.shape(angles_for_pmt)[0]
                #skipping pmts with no photon hits, and only 1 photon hit (in which case the variance will be undefined with ddof=1)
                if n_angles < 2:
                    continue
                norms = np.repeat(1.0, n_angles)
                mean_angle = normalize(np.mean(angles_for_pmt, axis=0))
                projection_norms = np.dot(angles_for_pmt, mean_angle)
                orthogonal_complements = np.sqrt(np.maximum(norms**2 - projection_norms**2, 0.))
                variance = np.var(orthogonal_complements, ddof=1)

                total_means[loops, i] = mean_angle
                total_variances[loops, i] = variance
                amount_of_hits[loops, i] = n_angles

            loops += 1
            if loops == nevents:
                break

         #combining all of the variances and mean_angles with the weightings for each event in order to create a single value for each pmt for the entire simulation. Masking the amount_of_hits first so that we don't divide by 0 when averaging for pmts that recieve no hits: CURRENTLY WE ARE SKIPPING EVERY PMT THAT ONLY HAS UP TO NEVENTS HITS: I.E. THE TOTAL DEGREE OF FREEDOM IN THE DIVISION FOR THE VARIANCE AVERAGING.
         bad_pmts = np.where(np.sum(amount_of_hits, axis=0) <= nevents)[0]
         
         # temporary, for debugging:
         n_hits = np.sum(amount_of_hits, axis=0)
         print n_hits
         print "PMTs w/ 0 hits: " + str(len(np.where(n_hits < 1)[0])*1.0/self.npmt_bins)
         print "PMTs w/ < nevents hits: " + str(len(np.where(n_hits < nevents)[0])*1.0/self.npmt_bins)
         print "PMTs w/ < 100 hits: " + str(len(np.where(n_hits < 100)[0])*1.0/self.npmt_bins)
 
         m_means = np.zeros_like(total_means)
         m_means[:, bad_pmts] = 1
         m_variances = np.zeros_like(total_variances)
         m_variances[:, bad_pmts] = 1

         masked_total_means = np.ma.masked_array(total_means, m_means)
         masked_total_variances = np.ma.masked_array(total_variances, m_variances)
        
         
         reshaped_weightings = np.reshape(np.repeat(amount_of_hits, 3), (nevents, self.npmt_bins, 3))
         averaged_means = np.ma.average(masked_total_means, axis=0, weights=reshaped_weightings)
         averaged_variances = np.ma.average(masked_total_variances, axis=0, weights=amount_of_hits - 1)

         #unmasking masked entries to turn them into 0.0s
         averaged_means.mask = np.ma.nomask
         averaged_variances.mask = np.ma.nomask

         #turning type: MaskedArray into type: npArray so that it can be written.
         self.means = -np.array(averaged_means.astype(np.float32)).T
         self.sigmas = np.sqrt(np.array(averaged_variances.astype(np.float32)))
                
    def write_to_ROOT(self, filename):
        # Write the mean angles and sigmas to a ROOT file
        writer = GaussAngleRootWriter(filename)
        for i in range(self.npmt_bins):
            writer.write_PMT(self.means[:,i], self.sigmas[i], i)
        writer.close()

    def read_from_ROOT(self, filename):
        # Read the means and sigmas from a ROOT file
        self.is_calibrated = True
        reader = GaussAngleRootReader(filename)
        for bin_ind, mean, sigma in reader:
            self.means[:,bin_ind] = mean
            self.sigmas[bin_ind] = sigma
            if np.isnan(sigma):
                print "Nan read in for bin index " + str(bin_ind)
