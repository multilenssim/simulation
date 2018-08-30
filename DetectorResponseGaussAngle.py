from DetectorResponse import DetectorResponse
from chroma.transform import normalize
from chroma.event import Photons

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import time
import pickle
import h5py
import deepdish as dd

import linalg_3
from logger_lfd import logger

# Original is in chroma.transform
@jit(nopython=True)
def norm(x):
    "Returns the norm of the vector `x`."
    return np.sqrt((x*x).sum(-1))

# Dot product of a 2d array with a 1d array
@jit(nopython=True)
def my_dot(array1, array2):
    result = np.empty(len(array1), dtype=np.float64)      # May not need 64?
    i = 0
    for i in range(len(array1)):
        result[i] = linalg_3.dot(array1[i], array2)
        i += 1
    return result

def assign_photons(npmt_bins, n_det, pmt_bins):
    start = time.time()
    pmt_photons = np.empty(npmt_bins, dtype=list)
    for photon in range(n_det):
        if photon % 1000000 == 0:
            logger.info("Photon " + str(photon) + " of " + str(n_det) + ': ' + str(time.time() - start))
        pmt = int(pmt_bins[photon])
        if pmt_photons[pmt] is None:
            pmt_photons[pmt] = [photon]
        else:
            pmt_photons[pmt] += [photon]
    return pmt_photons

@jit(nopython=True)
def compute_pmt_calibration(angles_for_pmt, n_min):
    mean = np.array([np.mean(angles_for_pmt[:,0]), np.mean(angles_for_pmt[:,1]), np.mean(angles_for_pmt[:,2])])
    mean_angle = mean / norm(mean)

    # For each PMT, get a pair of axes which form an
    # orthonormal coordinate system with the PMT mean direction
    u_dir = linalg_3.cross(mean_angle,np.array([0,0,1]))
    if not (np.dot(u_dir, u_dir) > 0): # In case mean_angle = [0,0,1]
        u_dir = linalg_3.cross(mean_angle,np.array([0,1,0]))
    u_dir = u_dir / norm(u_dir)
    v_dir = linalg_3.cross(mean_angle, u_dir)
    u_proj = my_dot(angles_for_pmt, u_dir)

    u_var = np.var(u_proj)
    v_proj = my_dot(angles_for_pmt, v_dir)
    v_var = np.var(v_proj)
    variance = (u_var+v_var)/2.

    return mean_angle, variance, u_var - v_var

class DetectorResponseGaussAngle(DetectorResponse):
    '''Detector calibration information is stored in Gaussian cones for each PMT: 
    the cones are represented as mean angles in 3D space and their uncertainties 
    (sigma for a cone of unit length) for the light hitting that PMT.
    '''
    def __init__(self, config, detectorxbins=10, detectorybins=10, detectorzbins=10, infile=None):
        # If passed infile, will automatically read in the calibrated detector mean angles/sigmas
        DetectorResponse.__init__(self, config, detectorxbins, detectorybins, detectorzbins)
        self.means = np.zeros((3,self.npmt_bins))
        self.sigmas = np.zeros(self.npmt_bins)
        if infile is not None:
            logger.info('Creating detector response / calibration with: %s' % infile)
            if infile.endswith('.h5'):
                try:
                    self.read_from_hdf5(infile)
                except IOError as error:
                    logger.warning('No calibration file found %s.  Continuing uncalibrated' % infile)
                    return
                if self.config.uuid != self.config_in_cal_file.uuid:
                    logger.critical('UUID from calibration file does not match configuration: %s %s' % (self.config.uuid, self.config_in_cal_file.uuid))
                    exit(-1)
                    # TODO: Raise an exception
                else:
                    logger.info('Calibration file UUID matches')
            else:
                self.read_from_ROOT(infile)
        else:
            logger.warning('No calibration file specified.  Continuing uncalibrated')

    def _find_photons_for_pmt(self, photons_beg_pos, photons_end_pos, detected, end_direction_array, n_det, max_storage):
        beginning_photons = photons_beg_pos[detected]       # Include reflected photons
        ending_photons = photons_end_pos[detected]
        length = len(ending_photons)

        pmt_b = self.find_pmt_bin_array(ending_photons)   # This line is now duplicative
        end_point = self.lens_centers[pmt_b/self.n_pmts_per_surf]
        end_dir = normalize(end_point-beginning_photons)

        if n_det+length > max_storage:
            logger.critical("Too many photons to store in memory; not reading any further events.")
            return None
        end_direction_array[n_det:(n_det+length),:] = end_dir
        return ending_photons, length

    '''
    The calibration is performed in three steps:
    1. Loop over events to find the pmt that each photon hit, and the direction for each photon.  Produces file 'hits-config.pickle'
    2. Loop over all photons to gather them by pmt.  Produces file 'pmt-bins.config.pickle'
    3. Loop over all photons for each pmt to compute statistics for each pmt
    The files are only meant to preserve the intermediate state.  They are not required.
    '''
    def calibrate(self, simname, directory=".", nevents=-1, fast_calibration=False):
        # Use with a simulation file 'simname' to calibrate the detector
        # Creates a list of mean angles and their uncertainties (sigma for
        # a cone of unit length), one for each PMT
        # There are 1000 events in a typical simulation.
        # Uses all photons hitting a given PMT at once (better estimate of sigma,
        # but may run out of memory in some cases).
        # Will not calibrate PMTs with <n_min hits
        logger.info('Fast calibration: %s' % str(fast_calibration))
        self.is_calibrated = True
        start_time = time.time()

        base_hits_file_name = self.configname + '-hits'
        pickle_name = base_hits_file_name + '.pickle'
        hit_file_exists = False
        n_min = 10 # Do not calibrate a PMT if <n_min photons hit it
        try:
            logger.info('Attempting to load pickle hits file: %s%s' % (directory, pickle_name));      # TODO: This generally won't do anything because we no longer write this file
            with open(directory + pickle_name, 'rb') as inf:
                pmt_hits = pickle.load(inf)
            logger.info('Hit map pickle file loaded: ' + pickle_name)
            pmt_bins = pmt_hits['pmt_bins']
            end_direction_array = pmt_hits['end_direction_array']
            n_det = len(pmt_bins)
            hit_file_exists = True
        except IOError as error:
            # Assume file not found
            logger.info('Hit map pickle file not found.  Creating: ' + base_hits_file_name)

            using_h5 = False
            if simname.endswith('.h5'):
                using_h5 = True
                ev_file = dd.io.load(simname)
                # TODO: Check the UUID!!
                events_in_file = len(ev_file['photons_start'])
            else:
                from ShortIO.root_short import GaussAngleRootWriter, GaussAngleRootReader, ShortRootReader
                reader = ShortRootReader(simname)
                events_in_file = len(reader)
            logger.info('Loaded simulation file: %s' % simname)
            logger.info('Simulation event count: %d' % events_in_file)

            if nevents < 1:
                nevents = events_in_file

            max_storage = min(nevents*1000000,120000000) #600M is too much, 400M is OK (for np.float32; using 300M)
            end_direction_array = np.empty((max_storage,3),dtype=np.float32)
            pmt_bins = np.empty(max_storage,dtype=np.int)
            n_det = 0

            # Loop through events, store for each photon the index of the PMT it hit (pmt_bins)
            # and the direction pointing back to its origin (end_direction_array)
            loops = 0
            event_source = ev_file['photons_start'] if using_h5 else reader
            for index, ev_proxy in enumerate(event_source):  # Not sure if we can enumerate a reader????
                # TODO: This needs to be cleaned up
                if (using_h5):
                    photons_beg = Photons(ev_proxy, [], [], [])
                    photons_end = Photons(ev_file['photons_stop'][index], [], [], [], flags=ev_file['photon_flags'][index])
                else:
                    photons_beg = ev_proxy.photons_beg
                    photons_end = ev_proxy.photons_end

                loops += 1
                if loops > nevents:
                    break

                if loops % 100 == 0:
                    logger.info("Event " + str(loops) + " of " + str(nevents))
                    logger.handlers[0].flush()

                detected = (photons_end.flags & (0x1 <<2)).astype(bool)
                '''
                reflected_diffuse = (ev.photons_end.flags & (0x1 << 5)).astype(bool)
                reflected_specular = (ev.photons_end.flags & (0x1 << 6)).astype(bool)
                logger.info("Total detected: " + str(sum(detected * 1)))
                logger.info("Total reflected: " + str(sum(reflected_diffuse * 1) + sum(reflected_specular * 1)))
                good_photons = detected & np.logical_not(reflected_diffuse) & np.logical_not(reflected_specular)
                logger.info("Total detected and not reflected: " + str(sum(good_photons * 1)))
                '''
                if fast_calibration: 
                    ending_photons, length = self._find_photons_for_pmt(photons_beg.pos, photons_end.pos, detected,
                                                                       end_direction_array, n_det, max_storage)
                    pmt_b = self.find_pmt_bin_array(ending_photons)
                    if length is None:
                        break
                else:
                    beginning_photons = photons_beg.pos[detected] # Include reflected photons
                    ending_photons = photons_end.pos[detected]
                    length = np.shape(ending_photons)[0]
                    pmt_b = self.find_pmt_bin_array(ending_photons)
                    end_point = self.lens_centers[pmt_b/self.n_pmts_per_surf]
                    end_dir = normalize(end_point-beginning_photons)
                    # if end_direction_array is None:
                    #     end_direction_array = end_dir
                    # else:
                    #     end_direction_array = np.vstack((end_direction_array, end_dir))
                    #end_direction_array.append(end_dir)
                    if n_det+length > max_storage:
                        logger.info('Too many photons to store in memory; not reading any further events.')
                        break
                    end_direction_array[n_det:(n_det+length),:] = end_dir
                    # if pmt_bins is None:
                    #     pmt_bins = pmt_b
                    # else:
                    #     pmt_bins = np.hstack((pmt_bins, pmt_b))
                    #pmt_bins.append(pmt_b)
                pmt_bins[n_det:(n_det+length)] = pmt_b
                n_det += length
                if loops % 100 == 0:
                    logger.info('Photons detected so far: ' + str(n_det + length))
                    # logger.info('Sample pmt bins: ' + str(pmt_bins[n_det:(n_det+length)]))
                    logger.info("Time: " + str(time.time() - start_time))

        total_means = np.zeros((self.npmt_bins, 3))
        total_variances = np.zeros((self.npmt_bins))
        total_u_minus_v = np.zeros((self.npmt_bins))
        amount_of_hits = np.zeros((self.npmt_bins))

        end_direction_array.resize((n_det,3))
        logger.info("Time: " + str(time.time() - start_time))
        pmt_bins.resize(n_det)

        if not hit_file_exists:
			# TODO: No longer writing the pickle file
            # Write the pickle hits file regardless of whether using fast_calibration or not
            #pmt_hits = {'pmt_bins': pmt_bins, 'end_direction_array': end_direction_array}
            #with open(directory+pickle_name, 'wb') as outf:
            #    pickle.dump(pmt_hits, outf)
            with h5py.File(directory + base_hits_file_name + '.h5', 'w') as h5file:
                _ = h5file.create_dataset('pmt_bins', data=pmt_bins, chunks=True)   # TODO: Should we assign max shape?
                _ = h5file.create_dataset('end_direction_array', data=end_direction_array, chunks=True)   # TODO: Should we assign max shape?
            logger.info('Hit map file created: ' + pickle_name)

        logger.info("Finished collecting photons (or loading photon hit list).  Time: " + str(time.time()-start_time))

        if fast_calibration:
            bins_base_file_name = self.configname + '-pmt-bins'
            bins_pickle_file = bins_base_file_name + '.pickle'
            try:
                with open(directory + bins_pickle_file, 'rb') as inf:       # TODO: This generally won't do anything because we no longer write this file
                    pmt_photons = pickle.load(inf)
                logger.info('PMT photon list pickle file loaded: ' + bins_pickle_file)
            except IOError as error:
                start_assign = time.time()
                pmt_photons = assign_photons(self.npmt_bins, n_det, pmt_bins)
                logger.info("assign_photons took: " + str(time.time() - start_assign))
				# TODO: No longer writing the pickle file
                #with open(directory + bins_pickle_file, 'wb') as outf:
                #    pickle.dump(pmt_photons, outf)

				# TODO: Pure H5 does not work.  (Because?)
                #with h5py.File(directory + bins_file + '.h5', 'w') as h5file:
                #    _ = h5file.create_dataset('photon_pmts', data=pmt_photons, chunks=True)   # Should we assign max shape?
                #logger.info('Type: ' + str(type(pmt_photons)) + ' ' + str(type(pmt_photons[0])))
                dd.io.save(directory + bins_base_file_name + '.h5', pmt_photons)
                logger.info('PMT photon list file created: ' + bins_base_file_name + '.h5')

        logger.info("Finished listing photons by pmt.  Time: " + str(time.time() - start_time))

        draw_pmt_ind = -1

        # looping through each pmt in order to save a mean_angle and a variance
        for i in range(self.npmt_bins):
            if i % 10000 == 0:
                logger.info(str(i) + ' out of ' + str(self.npmt_bins) + ' PMTs')
                logger.handlers[0].flush()
                logger.info("Time: " + str(time.time()-start_time))

            if fast_calibration:
                photon_list = pmt_photons[i]
                angles_for_pmt = end_direction_array[photon_list]
                n_angles = len(angles_for_pmt)  # np.shape(angles_for_pmt)[0]
                # skipping pmts with <2 photon hits (in which case the variance will be undefined with ddof=1)
                # also skipping if <n_min photon hits
                if n_angles < 2 or n_angles < n_min:
                    logger.warning('Not enough angles for PMT: %d, %d' % (i, n_angles))
                    continue

                mean_angle, variance, uvvar = compute_pmt_calibration(angles_for_pmt, n_min)
            else:
                pmt_indices = np.where(pmt_bins == i)[0]
                if np.shape(pmt_indices)[0] == 0:
                    continue
                angles_for_pmt = end_direction_array[pmt_indices]

                n_angles = np.shape(angles_for_pmt)[0]
                #skipping pmts with <2 photon hits (in which case the variance will be undefined with ddof=1)
                #also skipping if <n_min photon hits
                if n_angles < 2 or n_angles<n_min:
                    continue
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
                norms = np.repeat(1.0, n_angles)
                projection_norms = np.dot(angles_for_pmt, mean_angle)
                orthogonal_complements = np.sqrt(np.maximum(norms**2 - projection_norms**2, 0.))
                #variance = np.var(orthogonal_complements, ddof=1)

                uvvar = u_var - v_var
            try:
                #draw_pmt_ind = None
                draw_pmt_ind = int(draw_pmt_ind)
                '''
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
                '''
            except ValueError:
                pass
            except TypeError:
                pass

            total_means[i] = mean_angle
            total_variances[i] = variance
            total_u_minus_v[i] = np.abs(uvvar)
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
        print "Mean U-V variance (abs): " + str(np.mean(total_u_minus_v))

        # Store final calibrated values
        self.means = -total_means.astype(np.float32).T
        self.sigmas = np.sqrt(total_variances.astype(np.float32))

    def calibrate_old(self, simname, nevents=-1):
        # Use with a simulation file 'simname' to calibrate the detector
        # Creates a list of mean angles and their uncertainties (sigma for
        # a cone of unit length), one for each PMT
        # There are 1000 events in a typical simulation.
        # Older method - estimates variance for each PMT one event at a time
        from ShortIO.root_short import ShortRootReader

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

        #combining all of the variances and mean_angles with the weightings for each event in order to create a single value for each pmt for the entire simulation.
        # Masking the amount_of_hits first so that we don't divide by 0 when averaging for pmts that recieve no hits:
        # CURRENTLY WE ARE SKIPPING EVERY PMT THAT ONLY HAS UP TO NEVENTS HITS: I.E. THE TOTAL DEGREE OF FREEDOM IN THE DIVISION FOR THE VARIANCE AVERAGING.
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
        from ShortIO.root_short import GaussAngleRootWriter

        # Write the mean angles and sigmas to a ROOT file
        writer = GaussAngleRootWriter(filename)
        for i in range(self.npmt_bins):
            writer.write_PMT(self.means[:,i], self.sigmas[i], i)
        writer.close()

    def read_from_ROOT(self, filename):
        from ShortIO.root_short import GaussAngleRootReader

        # Read the means and sigmas from a ROOT file
        logger.info('Loading root calibration file: %s' % filename)
        self.is_calibrated = True
        reader = GaussAngleRootReader(filename)
        for bin_ind, mean, sigma in reader:
            self.means[:,bin_ind] = mean
            self.sigmas[bin_ind] = sigma
            if np.isnan(sigma):
                print "Nan read in for bin index " + str(bin_ind)
        logger.info('Last bin_index: %d' % bin_ind)

    def read_from_hdf5(self, filename):
        calibration = dd.io.load(filename)
        self.means = calibration['means']
        self.sigmas = calibration['sigmas']
        self.is_calibrated = True
        self.config_in_cal_file = calibration['config']  # Long variable name to avoid overwriting config set in DetectorResponse.init()
