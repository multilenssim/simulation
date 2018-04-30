import copy
import numpy as np
from scipy import integrate, optimize, ndimage, spatial
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot
from chroma.transform import normalize
from chroma.sample import uniform_sphere

import time as time
from DetectorResponse import DetectorResponse      # Ick - some weird import loop due to detectorconfig importing utils
from DetectorResponseGAKW import DetectorResponseGAKW
from DetectorResponseGaussAngle import DetectorResponseGaussAngle
from Tracks import Tracks, Vertex

from logger_lfd import logger

import pprint

class EventAnalyzer(object):
    '''An EventAnalyzer has methods of reconstructing an event and gauging
    the performance of this reconstruction. There are multiple ways to do
    such reconstruction, but all require a calibrated DetectorResponse 
    object to do so.
    '''
    def __init__(self, det_res):
        if not isinstance(det_res, DetectorResponse):
            print "Warning! Passed argument det_res is not a valid DetectorResponse object."
        self.det_res = det_res
        self.set_style()
        
    def analyze_event_PDF(self, eventfile, event_pos=None):
        from ShortIO.root_short import ShortRootReader
        from DetectorResponsePDF import DetectorResponsePDF

        #takes an event and constructs the pdf based upon which pmts were hit.
        if not isinstance(self.det_res, DetectorResponsePDF):
            print "Detector response must be of the class DetectorResponsePDF to use this method! Exiting."
            return
        reader = ShortRootReader(eventfile)
        for ev in reader:
            detected = (ev.photons_end.flags & (0x1 <<2)).astype(bool) 
            ending_photons = ev.photons_end.pos[detected]
            event_pmt_bin_array = self.det_res.find_pmt_bin_array(ending_photons)
            length = np.shape(ending_photons)[0]
            print "number of photons recording hits in event: " + str(length)

            final_pdf = np.zeros((self.det_res.detectorxbins, self.det_res.detectorybins, self.det_res.detectorzbins))
            f_squared = np.zeros((self.det_res.detectorxbins, self.det_res.detectorybins, self.det_res.detectorzbins))
            for i in range(length):
                event_pmt_bin = event_pmt_bin_array[i]
                if (event_pmt_bin == -1) or (self.det_res.pdfs[event_pmt_bin] == []):
                    continue
                else:
                    #final_pdf = np.add(final_pdf, self.det_res.pdfs[event_pmt_bin])
                    final_pdf += self.det_res.pdfs[event_pmt_bin]
                    f_squared += self.det_res.pdfs[event_pmt_bin]**2

            # Replace any value divided by 0 with 0, replace negative values with 0
            zero_mask = 1.0*(final_pdf == 0.)
            final_pdf_mask = np.ma.masked_array(final_pdf, zero_mask, fill_value=0.)
            f_squared_mask = np.ma.masked_array(f_squared, zero_mask, fill_value=0.)
            V_temp = np.array(f_squared_mask/final_pdf_mask)
            V_final = np.maximum(final_pdf - V_temp, 0.)
            V_final = np.float32(V_final/float(np.sum(V_final)))

            final_pdf = np.float32(final_pdf/float(np.sum(final_pdf)))
            
            # print "number of photons recording hits in event: " + str(length)
            # final_pdf.tofile('finalpdf-for-config-4-at-(6,6,6).txt')
            
            self.det_res.plot_pdf(final_pdf, '(adding model) PDF of ' + str(eventfile))
            self.det_res.plot_pdf(V_final, '(squared adding model) PDF of ' + str(eventfile))
            self.det_performance(final_pdf, 200)
            self.det_performance(V_final, 200)
            self.det_performance(final_pdf, 200, startingtype='mode')
            self.det_performance(final_pdf, 200, startingtype='center_of_mass')
            if not event_pos is None:
                self.det_performance(final_pdf, 200, startingtype='actual_position', event_location=event_pos)
        return final_pdf
    
    def analyse_perfect_res_event(self, eventfile, detector_dir_list_file, event_pos=None, recon_mode='angle', sig_cone=0.01, n_ph=0):
        '''Takes an event recorded without lenses and makes a final pdf of where the photons are most likely to have come from.
        Allows for different methods of calculating this, set with the recon_mode parameter. 
        If recon_mode='angle' (default), makes a list of ending directions for each pmt. A query is made to match a photon's direction with the 
        closest direction to it in the corresponding perfect_res_dir_list. This matches each photon hit for each pmt to a detector 
        that it is believed the photon came from. final_pdf then adds hits to these detector bins (for each pmt) in order to construct 
        the final pdf. 
        If recon_mode='cos', the method is the same as for 'angle' except the matching of photon angle to detector bin is done using the dot
        product between the two angles (really (1+a.b)/2, where a and b are the angles, to keep within [0,1]), so that all detector bins]
        have a continuous value added to them in the final pdf.
        For both 'cos' and 'angle', the quantity of detectorbins used in this file must match the quantity used in the detector_dir_list_file.  
        If recon_mode='cone', detector_dir_list_file is not needed, but sig_cone can be set. This method finds the angle between the each photon's 
        end direction and the vector from the center of the PMT it hit and a given detector bin. The probability of the photon coming from 
        that bin is then set by a Gaussian in angle (mean of 0, sigma given by sig_cone), so that bins near the photon's path are most likely,
        and the angular resolution can be adjusted by setting sig_cone. The probabilities across all bins are then normalized to one for each
        photon before adding to final_pdf (which is then normalized again after all photons are included).
        Setting n_ph>0 means only n_ph photons from the event file will be read in.
        '''

        from ShortIO.root_short import ShortRootReader, AngleRootReader

        # Check that a valid reconstruction mode was chosen
        if not (recon_mode=='angle' or recon_mode=='cos' or recon_mode=='cone'):
            print 'Unrecognized reconstruction mode! Exiting.'
            return None

        # Initialize detector_dir_list if using it
        if (recon_mode=='angle' or recon_mode=='cos'):
            detector_dir_list = [[] for i in range(self.det_res.npmt_bins)]
            first_reader = AngleRootReader(detector_dir_list_file)
            for bin_ind, angle in first_reader:
                detector_dir_list[bin_ind] = angle
        elif recon_mode=='cone': # Else we're in cone reconstruction mode
            print 'Cone reconstruction mode'
        
        ndetbins = self.det_res.detectorxbins*self.det_res.detectorybins*self.det_res.detectorzbins
        # Get list of bin positions
        bin_pos_array = self.det_res.bin_to_position_array()

        reader = ShortRootReader(eventfile)
        for ev in reader:
            detected = (ev.photons_end.flags & (0x1 <<2)).astype(bool) 
            beginning_photons = ev.photons_beg.pos[detected]
            ending_photons = ev.photons_end.pos[detected]
            end_direction_array = normalize(ending_photons-beginning_photons)
            length = np.shape(ending_photons)[0]
            if n_ph > 0: 
                length = n_ph
            print "number of photons recording hits in event: " + str(length)

            #creating an array of pmt bins for the ending_photons
            event_pmt_bin_array = self.det_res.find_pmt_bin_array(ending_photons)

            final_pdf = np.zeros((self.det_res.detectorxbins, self.det_res.detectorybins, self.det_res.detectorzbins))
            f_squared = np.zeros((self.det_res.detectorxbins, self.det_res.detectorybins, self.det_res.detectorzbins)) #Test
            nbad_pmt_bins = 0
            for i in range(length):
                photon_pdf = np.zeros((self.det_res.detectorxbins, self.det_res.detectorybins, self.det_res.detectorzbins))
                if i % 1000 == 0:
                    print str(i) + ' out of ' + str(length)
                event_pmt_bin = event_pmt_bin_array[i]
                end_direction = end_direction_array[i]

                # code for 'angle' and 'cos' recon_mode
                if (recon_mode=='angle' or recon_mode=='cos'):
                    if np.shape(detector_dir_list[event_pmt_bin]) != (ndetbins, 3):
                        nbad_pmt_bins += 1
                        continue

                    detector_array = detector_dir_list[event_pmt_bin]

                    if recon_mode=='cos':
                        cosines = np.dot(detector_array, np.transpose(end_direction))
                        values = (1+cosines)/2.0
                        for i in range(ndetbins):
                            if np.sum(detector_array[i]) == -15:
                                continue
                            else:
                                xbin, ybin, zbin = self.det_res.detectorbin_index_to_tuple(i, self.det_res.detectorxbins, self.det_res.detectorybins, self.det_res.detectorzbins)
                                photon_pdf[xbin, ybin, zbin] += values[i]
                        photon_pdf = np.float32(photon_pdf/float(np.sum(photon_pdf)))
                        final_pdf += photon_pdf
                        f_squared += photon_pdf**2
                                
                        
                    elif recon_mode=='angle':
                        det_tree = spatial.cKDTree(detector_array)
                        nearest_detector_index = det_tree.query(end_direction)[1]
                        if detector_array[nearest_detector_index][0] < -1:
                            print "bad angle"
                            continue

                        xbin, ybin, zbin = self.det_res.detectorbin_index_to_tuple(nearest_detector_index, self.det_res.detectorxbins, self.det_res.detectorybins, self.det_res.detectorzbins)

                        final_pdf[xbin, ybin, zbin] += 1
                        f_squared[xbin, ybin, zbin] += 1

                # code for 'cone' recon_mode
                elif recon_mode=='cone':
                    pmt_pos = self.det_res.pmt_bin_to_position(event_pmt_bin) # Get hit PMT position
                    # Get angles from photon track to vector from PMT to each detector bin
                    angles = self.get_angles(pmt_pos, bin_pos_array, -end_direction) 
                    pdf_temp = self.gauss_prob(angles, sig_cone)
                    pdf_temp = np.reshape(pdf_temp, np.shape(final_pdf))
                    # Add this photon's pdf to the final one
                    final_pdf += pdf_temp
                    f_squared += pdf_temp**2
                    #print pdf_temp[30:35,30:35,30:35]

                    if n_ph > 0: # Few photon debugging
                        print 'Hit PMT position: ' + str(pmt_pos)
                        print 'Photon direction: ' + str(end_direction)
                        #print 'final pdf: ' + str(final_pdf)
            
            print 'Bad pmt bins: ' + str(nbad_pmt_bins)

            # Replace any value divided by 0 with 0, replace negative values with 0
            zero_mask = 1.0*(final_pdf == 0.)
            final_pdf_mask = np.ma.masked_array(final_pdf, zero_mask, fill_value=0.)
            f_squared_mask = np.ma.masked_array(f_squared, zero_mask, fill_value=0.)
            V_temp = np.array(f_squared_mask/final_pdf_mask)
            V_final = np.maximum(final_pdf - V_temp, 0.)
            V_final = np.float32(V_final/float(np.sum(V_final)))

            final_pdf = np.float32(final_pdf/float(np.sum(final_pdf)))
            #print final_pdf
            
            if n_ph > 0:
                ending_photons = ending_photons[:n_ph,:]
                self.det_res.plot_pdf(final_pdf, 'Perfect Resolution PDF of ' + str(eventfile), beginning_photons, ending_photons, bin_pos_array)
                self.det_res.plot_pdf(V_final, 'Perfect Resolution PDF of ' + str(eventfile), beginning_photons, ending_photons, bin_pos_array)
            else:
                self.det_res.plot_pdf(final_pdf, 'Perfect Resolution PDF of ' + str(eventfile), bin_pos=bin_pos_array)                
            #self.det_performance(final_pdf, 200, startingtype='max_starting_zone')
            self.det_performance(final_pdf, 200, startingtype='mode')
            self.det_performance(V_final, 200, startingtype='mode')
            self.det_performance(final_pdf, 200, startingtype='center_of_mass')
            if not event_pos is None:
                self.det_performance(final_pdf, 200, startingtype='actual_position', event_location=event_pos)
        return final_pdf
        
    def analyze_one_event_AVF(self, event, sig_cone=0.01, n_ph=0, min_tracks=4, chiC=3., temps=[256, 0.25], tol=1.0, debug=False, lens_dia=None):
        # Takes a Chroma Event object, turns the detected photons into Track objects, and performs the adaptive
        # vertex fitting algorithm to find a list of Vertex objects for this event
        # If self.det_res is not calibrated or of the wrong type, the Track objects will be constructed using
        # the true directions the photons came from
        
        # Create a list of tracks (from hits and detector response, if calibrated, else w/ perfect resolution)
        tracks = self.generate_tracks(event, heat_map=False,sig_cone=sig_cone, n_ph=n_ph, lens_dia=None,debug=False)
        
        # Find/fit vertices using adaptive vertex fitter method
        return self.AVF(tracks, min_tracks, chiC, temps, tol, debug) # list of Vertex objects
            
    def AVF(self, tracks, min_tracks=4, chiC=3., temps=[256, 0.25], tol=1.0, debug=False):
        '''Adaptive vertex fitter algorithm to find vertex locations/uncertainties and which tracks 
        to associate with each vertex.
        The algorithm, in brief:
        While there are fewer than min_tracks unassociated tracks:
            Seed a vertex position by adding the remaining tracks and finding the max location
            Iteratively adjust the vertex position to reduce the weighted sum of squared distances
            from tracks to the vertex, where the weight is a time-dependent sigmoid function with
            inlier threshold given by chiC (in sigma) and annealing schedule given by
            T(m)=1+r*(T(m-1)-1), with T(0)=temps[0], r=temps[1] (if len(temps)>2, just use it)
            Continues iteration until position shifts by less than tol and temperature in annealing 
            schedule has stabilized. 
            Tracks with weight >0.5 are associated to the resulting vertex, and remaining tracks
            are fed back to the algorithm to find the next vertex.
            If min_tracks<1, use as a fraction of the total event's tracks.
        '''
        WEIGHT_CUT = 0.50

        # Get an array of voxel positions within the detector, for repeated use
        bin_pos_array = np.array(self.det_res.bin_to_position_array())  # Returns 10x10x10 = 1000 coordinate positions
        
        # Sets how much to scale down changes in vertex position by, should they fail to improve fit
        # Only used for analytic (matrix) solution, not numpy's fmin() optimization function
        vtx_scale_factor = 0.5

        # Sets whether to use Gauss probability (additive) or NLL (additive, equivalent to multiplying probs)
        doNLL = False

        vtcs = [] # List of vertices found
        tracks0 = copy.deepcopy(tracks) # Keep track of tracks - such meta
        
        # Find vertices iteratively until insufficient tracks remain
        min_global = 10 # No matter what, vertices must have this many tracks
        if min_tracks < 1.0: # Use fraction of total event's tracks
            min_tracks = int(min_tracks*float(len(tracks)))
        min_tracks = max(min_global, min_tracks)
        while len(tracks) >= min_tracks:
            # Find initial vertex location - get gaussian probabilities at each of the detector locations
            # Repurpose get_angles and gauss_prob to operate on all tracks at once? 
            final_pdf = np.zeros((self.det_res.detectorxbins, self.det_res.detectorybins, self.det_res.detectorzbins))
            f_squared = np.zeros((self.det_res.detectorxbins, self.det_res.detectorybins, self.det_res.detectorzbins))
 
            t0 = time.time()
            for ii, (hit_pos, mean, sig) in enumerate(tracks):
                # print hit_pos
                # print mean
                # print sig
                # print 'bin_pos shape: ' + str(np.shape(bin_pos_array))
                #if ii > 10:
					#break
                
                angles = np.reshape(self.get_angles(hit_pos, bin_pos_array, mean),np.shape(final_pdf))
                #print 'angles shape: ' + str(np.shape(angles))
                dists_scaled = np.tan(angles) # scaled so that projection of vector onto mean angle is 1
                #print 'dists scaled shape: ' + str(np.shape(dists_scaled)) # sig is (n,)
                if np.isnan(sig):
                    logger.info('Track sig is nan - skipping.')
                if doNLL:
                    gp = self.gauss_nll(dists_scaled, sig)
                else:
                    gp = self.gauss_prob(dists_scaled, sig)
                final_pdf += gp
                #final_pdf *= gp
                #f_squared += gp**2
                if False: # debug:
                    if ii % 1000 == 0:
                        plotpdf = final_pdf
                        plotpdf = np.float32(plotpdf / float(np.sum(plotpdf)))  # normalize, for plotting purposes

                        fig = self.det_res.plot_pdf(plotpdf, "Initial vtx position finding", bin_pos=bin_pos_array, show=False)
                        ax = fig.gca()
                        # ax.scatter(v_pos_max[0], v_pos_max[1], v_pos_max[2], color='green')
                        plt.figure(fig.number)
                        plt.show()
            # Replace any value divided by 0 with 0, replace negative values with 0
            #zero_mask = 1.0*(final_pdf == 0.)
            #final_pdf_mask = np.ma.masked_array(final_pdf, zero_mask, fill_value=0.)
            #f_squared_mask = np.ma.masked_array(f_squared, zero_mask, fill_value=0.)
            #V_temp = np.array(f_squared_mask/final_pdf_mask)
            #V_final = np.maximum(final_pdf - V_temp, 0.)
            #V_final = np.float32(V_final/float(np.sum(V_final)))

            #print np.sum(final_pdf)
            #print np.argmax(final_pdf), np.max(final_pdf)
            #print bin_pos_array[np.argmax(final_pdf)]
            #final_pdf = np.float32(final_pdf/float(np.sum(final_pdf))) #normalize, for plotting purposes

            t1 = time.time()
            #print "Initial vtx location time: " + str(t1-t0)
            max_bin = np.argmax(final_pdf)
            #max_bin = np.argmax(V_final)
            v_pos_max = bin_pos_array[max_bin]
            #v_pos_max = np.array([0.,0.,0.]) # Could consider using this for n_ph>100...

            # # Get center of mass of neighboring bins to max_bin to use as initial vtx pos
            # shape = np.shape(final_pdf)
            # xx, yy, zz = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
            # bin_array = zip(xx.ravel(), yy.ravel(), zz.ravel())
            # max_bintup = np.unravel_index(max_bin,np.shape(final_pdf))
            # max_bin_com, _ = self.get_com(final_pdf, max_bintup)
            # if max_bin_com is not None: # Stick with original position if center of mass calc fails
            #     v_pos_max = np.array(self.det_res.bin_to_position(max_bin_com))
            #max_bin = self.find_max_starting_bin(bin_array, final_pdf, np.shape(final_pdf))

            # Force the initial starting position
            #v_pos_max = np.asarray([1000., 0., 0.])
            logger.info('Initial vertex position: ' + str(v_pos_max))
            if debug:
                self.plot_tracks(tracks,highlight_pt=v_pos_max)
                if False:    # Plot the grid of closest approach hits
                    if doNLL:
                        plotpdf = 1./final_pdf # Take inverse if using NLL method (since values close to 0 are better)
                    else:
                        plotpdf = final_pdf
                    plotpdf = np.float32(plotpdf/float(np.sum(plotpdf))) #normalize, for plotting purposes

                    fig=self.det_res.plot_pdf(plotpdf, "Initial vtx position finding", bin_pos=bin_pos_array, show=False)
                    #plt.show()
                    ax = fig.gca()
                    ax.scatter(v_pos_max[0],v_pos_max[1],v_pos_max[2],color='green')

                    # codethis = raw_input('Do some code tweaks>')
                    # while codethis not in ['','q','exit'] :
                    #     exec(codethis)
                    #     codethis = raw_input()

                    plt.figure(fig.number)
                    plt.show()
                    #raw_input("Waiting for input")
                    #plt.show(fig)

            # Temporary, for testing; try AVF with several different initial locations near that found above
            # Pick from a uniform sphere of radius rad_frac*inscribed_radius
            n_pos0 = 0
            rad_frac = 0.5
            if n_pos0 > 0:
                ddir0 = uniform_sphere(n_pos0)
                drad0 = rad_frac*self.det_res.inscribed_radius*np.random.uniform(0.0, 1.0, n_pos0)**(1.0/3)
                #print "drad0: " + str(drad0)
                dpos0 = drad0*ddir0.T
                pos0 = v_pos_max+dpos0.T
                v_pos0 = np.vstack((v_pos_max, pos0))
                #print [np.linalg.norm(v_pos0[ii,:]-v_pos_max) for ii in range(n_pos0+1)]
            else:
                n_pos0 = 0
                v_pos0 = np.array(v_pos_max,ndmin=2)
            
            vtx_list = []
            obj_list = []
            wt_list = []

            #### Finished determining starting vertex (or vertices - if debugging) ####
            
            # For each starting position, find vtx position and associated tracks; keep one with best obj
            for ii in range(n_pos0+1):
                v_pos = v_pos0[ii,:]            
                # Initialize this vertex
                logger.info('===============================')
                logger.info('=== AVF starting vertex: %s (%d)===' % (str(v_pos), ii))
                vtx = Vertex(v_pos, -1., 0)
            
                # # Temporary, for testing; just use initial vtx position
                # vtcs.append(vtx)
                # break
            
                v_pos_rec = [] # Record vtx position at each iteration
                wt_rec = [] # Record weights for all tracks at each iteration
                obj_rec = [] # Record objective function (weighted sum of squares) at each iteration
                dv_rec = [] # Record how far vtx moves at each iteration
                v_pos_old = v_pos
                dv = 2.0*tol
                dv_old = 2.0*tol
                mm = -1
                Tm = temps[0]

                first_iteration = True

                # Converges when: vtx position is stable to w/in tol for two iterations and temp has reached 1.1
                while dv > tol or dv_old > tol or Tm > 1.1: 
                    v_pos_rec.append(v_pos) # Record current vtx position
                    # Change annealing temperature; use list directly if not length 2, else use formula
                    mm += 1
                    if len(temps) != 2 or mm == 0:
                        if mm >= len(temps):
                            Tm = 1.0
                        else:
                            Tm = temps[mm]
                    else:
                        Tm = 1+temps[1]*(Tm-1)
                    
                    # Calculate closest points along the tracks to vtx, scaled sigmas, distances to vtx,
                    # distances in sigma units, weights, and objective function
                    r0, sig0, d0, chi0, wt0, obj0 = self.get_track_fit_params(tracks, vtx, chiC, Tm)

                    '''
                    if first_iteration:
                        _,bn,_ = plt.hist(d0,bins=100)
                        plt.yscale('log', nonposy='clip')
                        plt.xlabel('Distance from test vertex')
                        ##plt.show()
                        first_iteration = False
                    '''
                    wt_rec.append(wt0) # Record current weights
                    obj_rec.append(obj0) # Record current objective function value
                    '''
                    plt.hist(d0, bins=100)
                    plt.xlabel('Distance from test vertex')
                    plt.show()

                    plt.hist(wt0, bins=100)
                    plt.xlabel('Weights')
                    plt.show()
                    '''
                    if debug:
                        print_max = min(len(d0), 50)
                        print_max_pts = min(len(d0), 15)
                        print "New closest points: " + str(r0.T[:print_max_pts,:])
                        print "New distances: " + str(d0[:print_max])
                        print "New chi0: " + str(chi0[:print_max])
                        print "New wt0: " + str(wt0[:print_max])
                        print "Sig: " + str(sig0[:print_max])
                        #fig=self.det_res.plot_pdf(final_pdf, "Initial vtx position finding", bin_pos=bin_pos_array, show=False)
                        #self.plot_tracks(tracks,r0,v_pos)
                        # ax = fig.gca()
                        # ax.scatter(r0[0,:],r0[1,:],r0[2,:],color='red')
                        # print "Show new figure..."
                        # plt.show(fig)
                    
                    # # Intermediate matrix algebra, analytically minimizes weighted sum of squared distances
                    # # (in sigma), after linearizing distance formula by expanding around vtx position
                    # ai = (v_pos-r0.T).T/d0
                    # ap = wt0*ai/(sig**2)
                    # A = np.dot(ai,ap.T)
                    # B = -np.dot(ap,d0)

                    
                    # # Calculate change in vtx position, as the solution to A*delta_v = B
                    # delta_v = np.linalg.solve(A,B)
                    # v_pos = v_pos_old + delta_v
                    # vtx.pos = v_pos
                    
                    # # Checks if weighted sum of squares has improved (recalculate);
                    # # if not, scale down delta_v and repeat until it has
                    # r1, sig1, d1, chi1, wt1, obj1 = self.get_track_fit_params(tracks, vtx, chiC, Tm)

                    # scale_down = 1.0
                    # while obj1 > obj0:
                    #     scale_down *= vtx_scale_factor
                    #     v_pos = v_pos_old + delta_v*scale_down
                    #     vtx.pos = v_pos
                    #     r1, sig1, d1, chi1, wt1, obj1 = self.get_track_fit_params(tracks, vtx, chiC, Tm)
                    #     if debug:
                    #         print "Objective function failed to improve." 
                    #         print "Old value: " + str(obj0) + ", New value: " + str(obj1)
                    
                    # Alternate method: find minimum numerically using scipy
                    def obj_func(vtx_pos):
                        vtx.pos = vtx_pos
                        _, _, _, _, _, obj = self.get_track_fit_params(tracks, vtx, chiC, Tm)
                        return obj

                    # Minimizes obj_func; can set tolerance xtol and ftol but defaults of 1e-4 are good
                    # disp=True will show convergence info
                    # Check for improvement in objective?
                    v_opt = optimize.fmin(obj_func, v_pos, disp=False)
                    vtx.pos = v_opt
                    _, _, _, _, _, obj1 = self.get_track_fit_params(tracks, vtx, chiC, Tm)
                    # Retry w/ slightly different initial guess if outside detector or objective is worse
                    n_tries = 1
                    max_tries = 10
                    while np.linalg.norm(v_opt) > self.det_res.inscribed_radius and obj1 > obj0 and n_tries < max_tries: 
                        n_tries += 1
                        logger.warning('Vertex placed outside of detector, or objective failed to improve - trying again, try '+str(n_tries))
                        ddir_opt = uniform_sphere()
                        # Shift by a distance of up to 10% of the inscribed radius, in a random direction
                        drad_opt = 0.1*self.det_res.inscribed_radius*np.random.uniform(0.0, 1.0, 1)**(1.0/3)
                        dv_opt = drad_opt*ddir_opt.T
                        v_new = v_pos+dv_opt.T
                        v_opt = optimize.fmin(obj_func, v_new, disp=False)
                        vtx.pos = v_opt
                        _, _, _, _, _, obj1 = self.get_track_fit_params(tracks, vtx, chiC, Tm)

                    v_pos = v_opt    
                    
                    dv_old = dv
                    dv = np.linalg.norm(v_pos - v_pos_old) # Change in vtx position
                    dv_rec.append(dv)
                    if debug:
                        print "New vtx position: " + str(v_pos)
                        # print "ai: " + str(ai)
                        # print "ai^2: " + str(ai**2)
                        # print "ap: " + str(ap)
                        # print "A: " + str(A)
                        # print "B: " + str(B)
                        #print "dV: " + str(np.dot(np.linalg.inv(A),B))
                        print "dV: " + str(dv)
                        #print "objective: " + str(obj1)
                        print "Temp this iteration: " + str(Tm)
                    
                    v_pos_old = v_pos

                # Done finding this vertex position                
                # TODO: calculate error
                # TODO: check that final associated tracks are at least min_tracks, else break
                vtx_ph = np.sum(1.0*(wt0 >= WEIGHT_CUT))
                logger.info('Associated tracks: %d' % int(vtx_ph))
                '''
                _,bn,_ = plt.hist(wt0,bins=100)
                plt.yscale('log', nonposy='clip')
                plt.xlabel('Track weights')
                plt.show()
                '''
                vtx.n_ph = vtx_ph
                vtx_list.append(vtx)
                obj_list.append(obj0)
                wt_list.append(wt0)

                logger.info('=== AVF finished vertex: %s (%d), result: %s ===' % (str(v_pos0[ii, :]), ii, str(vtx.pos)))
                logger.info('===============================')
                if debug:
                    print "Position record: " + str(v_pos_rec)
                    print "dv record: " + str(dv_rec)
                    #print "Weight record: " + str(wt_rec)
                    print "Objective function record: " + str(obj_rec)
                    logger.info('Tracks associated with this vertex: %d' % len(tracks))
                    # Make plot of vtx pos vs iteration, weights and obj function vs iteration
                    self.plot_tracks(tracks,path=np.array(v_pos_rec).T)
                    #self.plot_weights(np.array(wt_rec),obj=np.array(obj_rec))
                    #self.plot_weights(np.random.random_sample(np.shape(np.array(wt_rec))),obj=np.array(obj_rec))

            # print [vt.pos for vt in vtx_list]
            # print v_pos_max
            #print [np.linalg.norm(vt.pos-vtx_list[0].pos) for vt in vtx_list]
            # print obj_list
            # Choose best vertex from among different initial positions
            vtx_pos_ar = np.array([vt.pos for vt in vtx_list]) # Vertices found w/ different initial pos
            vtx_pos_stdcoord = np.std(vtx_pos_ar,axis=0)
            #print np.sqrt(np.sum(vtx_pos_stdcoord**2))
            #print obj_list
            opt_ind = np.argmin(obj_list)
            vtx = vtx_list[opt_ind] # This is the best reconstructed vertex for this iteration         

            # Record tracks associated to this vertex
            trx_assoc = copy.deepcopy(tracks) # Tracks associated to vtx
            trx_assoc.cull(np.nonzero(wt_list[opt_ind]>=WEIGHT_CUT))

            # Cull tracks to only those tracks which were not already associated
            #tracks.cull(np.nonzero(wt_list[opt_ind]>1.1)) # Use this to stop after 1st vtx is found
            tracks.cull(np.nonzero(wt_list[opt_ind]<WEIGHT_CUT))
            logger.info('>>>>>>>> Remaining tracks: %d' % len(tracks))

            # Check that vertex has a (normalized) objective function less than qual*chiC
            qual = 0.5
            #qual = 2 
            _, _, _, _, _, obj_fin = self.get_track_fit_params(trx_assoc, vtx, chiC, 1.0)
            if debug:
                print 'obj_fin (uses only associated tracks): %f / %f' % (obj_fin, qual*chiC)
                print "Associated tracks: " + str(len(trx_assoc))
            if obj_fin < qual*chiC and len(trx_assoc) >= min_tracks:
                vtx.err = obj_fin # use objective function to judge quality of vertex
                vtx.n_ph = len(trx_assoc)
                vtcs.append(vtx)
            else: # If the vertex quality is poor or has too few tracks, stop looking for more vertices
                logger.info('Vertex quality too poor / too few tracks: %f, Targets: %f %d. Dropping and quitting.' % (obj_fin, qual*chiC, min_tracks))
                break
        
        # Restrict to unique vertices (distance > tol; only position is relevant for this step)
        vtcs_unique = []
        skip = []
        for ii in range(len(vtcs)):
            if ii in skip:
                continue
            vtx1 = vtcs[ii]
            vtcs_unique.append(vtx1)
            for jj in range(ii+1, len(vtcs)):
                if jj in skip:
                    continue
                vtx2 = vtcs[jj]
                if np.linalg.norm(vtx1.pos-vtx2.pos) < tol: # If two vertices are w/in tol, ignore second vtx
                    skip.append(jj)

        if debug:
            print "Total vertices: " + str(len(vtcs))
            print "Unique vertices: " + str(len(vtcs_unique))

        vtcs = vtcs_unique

        if debug:
            print "Vertices before reassociation: "
            for vtx in vtcs:
                print "Vtx pos: " + str(vtx.pos)
                print "Vtx n_ph: " + str(vtx.n_ph)
                print "Vtx quality: " + str(vtx.err)

        # Redo association: for each track, associate to closest vertex if it passes the chiC cut
        # First, calculate distances
        dists = np.zeros((len(vtcs),len(tracks0)))
        for ii, vtx in enumerate(vtcs):
            r, sig = tracks0.closest_pts_sigmas(vtx)
            dists[ii,:] = vtx.dist(r)/sig # distance (in sigma) from tracks to current vtx

        # Find index of closest vertex for each track
        if len(dists) == 0:
            logger.info('AVF: no vertices found')
            return None

        vtx_closest = np.argmin(dists, axis=0)
        
        # Associate tracks and recalculate n_ph, err
        for ind, vtx in enumerate(vtcs):
            tracks_assoc = copy.deepcopy(tracks0) 
            # Keep only tracks closest to this vtx, with distance (in sigma) less than chiC
            #tracks_assoc.cull(np.nonzero(np.logical_and(vtx_closest==ind, dists[ind,:] < chiC))) 
            # Keep only tracks closest to this vtx
            tracks_assoc.cull(np.nonzero(vtx_closest==ind)) 
            vtx.tracks = tracks_assoc # Record the tracks for this vertex
            vtx.n_ph = len(tracks_assoc)
            _, _, _, _, _, obj_assoc = self.get_track_fit_params(tracks_assoc, vtx, chiC, 1.0)
            vtx.err = obj_assoc # use objective function to judge quality of vertex
        

        if not vtcs: # len(tracks) < min_tracks:
            print "Warning! No vertices found."   
        if debug:
            print "Vertices after reassociation: "
            for vtx in vtcs:
                print "Vtx pos: " + str(vtx.pos)
                print "Vtx n_ph: " + str(vtx.n_ph)
                print "Vtx quality: " + str(vtx.err)
            
        return vtcs

    def QE(self,gen,qe):
        #applies a given quantum efficiency to each pixel to a vector containing the hit pixel
        mask = []
        for e in np.unique(gen):
            arr_id = np.where(gen==e)[0]
            counts = np.random.poisson(qe*len(arr_id))
            if counts>len(arr_id):
                counts = np.random.choice(len(arr_id))
            fltr = np.random.choice(arr_id,counts,replace=False)
            mask.extend(fltr)
        return sorted(mask)

    '''
    # This is likely DEPRECATED already
    # Makes tracks from a list of Photon objects at the ending position
    # Returns 'enhanced' Track objects with lens/PMT positions included
    def generate_tracks_from_hit_positions(self, photons_end, debug=False):
        detected = (photons_end.flags & (0x1 << 2)).astype(bool)
        detected_photons = photons_end.pos[detected]
        logger.info('Using ' + str(len(detected_photons)) + ' detected of ' + str(len(photons_end)) + ' photons')

        event_pmt_bin_array = np.array(self.det_res.find_pmt_bin_array(detected_photons))  # Get PMT hit location
        # event_pmt_pos_array = np.array(self.det_res.pmt_bin_to_position(event_pmt_bin_array)).T     # Not needed???

        event_lens_bin_array = np.array(event_pmt_bin_array / self.det_res.n_pmts_per_surf)
        event_lens_pos_array = np.array([self.det_res.lens_centers[x] for x in event_lens_bin_array]).T

        # If detector is not calibrated or not of the GaussAngle subclass, quit
        if not (self.det_res.is_calibrated and (isinstance(self.det_res, DetectorResponseGAKW) or isinstance(self.det_res, DetectorResponseGaussAngle))):
            logger.severe('== Detector not calibrated ==')
            return None

        # Detector is calibrated, use response to generate tracks
        tracks = Tracks(event_lens_pos_array,
                        self.det_res.means[:, event_pmt_bin_array],
                        self.det_res.sigmas[event_pmt_bin_array],
                        lenses=event_lens_bin_array,
                        rings=None,
                        pixels_in_ring=event_pmt_bin_array,  # Not right yet...
                        lens_rad=self.det_res.lens_rad)
        # tracks = Tracks(event_pmt_pos_array, self.det_res.means[:,event_pmt_bin_array], self.det_res.sigmas[event_pmt_bin_array], lens_rad = 0.0000001)
        tracks.cull(np.where(tracks.sigmas > 0.001))  # Remove tracks with zero uncertainty (not calibrated)
        if np.any(np.isnan(tracks.sigmas)):
            print "Nan tracks!! Removing."
            nan_tracks = np.where(np.isnan(tracks.sigmas))
            tracks.cull(nan_tracks)
        if debug:
            print "Tracks for calibrated PMTs: " + str(len(tracks))
        # tracks.cull(np.where(tracks.sigmas<0.2)) # Remove tracks with too large uncertainty
        # tracks.sigmas[:] = 0.054 # Temporary! Checking if setting all sigmas equal to each other helps or hurts
        return tracks
    '''

    def generate_tracks(self, ev, qe=None, heat_map=False, sig_cone=0.0001, n_ph=0, lens_dia=None, debug=True):
        #Makes tracks for event ev; allow for multiple track representations?
        detected = (ev.photons_end.flags & (0x1 <<2)).astype(bool)
        logger.info('Detected: ' + str(detected))
        logger.info(str(ev.photons_end.flags));
        reflected_diffuse = (ev.photons_end.flags & (0x1 <<5)).astype(bool)
        reflected_specular = (ev.photons_end.flags & (0x1 <<6)).astype(bool)
        good_photons = detected & np.logical_not(reflected_diffuse) & np.logical_not(reflected_specular)
             
        beginning_photons = ev.photons_beg.pos[detected] # Include reflected photons
        ending_photons = ev.photons_end.pos[detected]
        length = np.shape(ending_photons)[0]
        logger.info('Using ' + str(len(ending_photons)) + ' detected of ' + str(len(ending_photons)) + ' photons')

        if debug:
            print "Total detected photons in event: " + str(sum(detected*1))
            print "Total photons: " + str(np.shape(ev.photons_end.pos)[0])
            print "Total reflected: " + str(sum(reflected_diffuse*1)+sum(reflected_specular*1))
            print "Total detected and not reflected: " + str(sum(good_photons*1))
        if n_ph > 0 and n_ph < length: # If > 0, use only n_ph photons from the event, randomly chosen
            choice = np.random.permutation(length)[:n_ph]#np.array(range(n_ph))#
            beginning_photons = beginning_photons[choice]
            ending_photons = ending_photons[choice]
            # beginning_photons = beginning_photons[start:(start+n_ph),:]
            # ending_photons = ending_photons[start:(start+n_ph),:]
            length = n_ph

        end_direction_array = normalize(ending_photons-beginning_photons).T
        event_pmt_bin_array, lenses, rings, pixels = np.array(self.det_res.find_pmt_bin_array_new(ending_photons)) # Get PMT hit indices

        #qe=1 # Just a test
        if qe == None:
            pass
        else:
            mask = self.QE(event_pmt_bin_array,qe)
            event_pmt_bin_array = event_pmt_bin_array[mask]
            lenses = lenses[mask]
            rings = rings[mask]
            pixels = pixels[mask]
            logger.info('Mask count: %d' % len(mask))

        # print('PMT bins: ' + str(event_pmt_bin_array))
        event_pmt_pos_array = np.array(self.det_res.pmt_bin_to_position(event_pmt_bin_array)).T
        event_lens_bin_array = np.array(event_pmt_bin_array/self.det_res.n_pmts_per_surf)
        event_lens_pos_array = np.array([self.det_res.lens_centers[x] for x in event_lens_bin_array]).T
        
        # If detector is not calibrated or not of the GaussAngle subclass, use actual photon angles
        # plus Gaussian noise (two different models, depending on if lens_dia is given)
        if not (self.det_res.is_calibrated and (isinstance(self.det_res,DetectorResponseGAKW) or isinstance(self.det_res,DetectorResponseGaussAngle))):
            logger.info('Detector is not calibrated (or is not gaussian response)')
            # If lens_dia is not given, only include angular noise of sig_cone
            sigmas = np.zeros(length)
            if lens_dia is None:
                logger.info('No lens diameter.  Using sig_cone only: %f' % sig_cone)
                sigmas.fill(sig_cone)
                sig_th = sig_cone
                hit_pos = event_pmt_pos_array

            # If lens_dia is given, include noise on the hit position; angular noise is set by lens_dia and
            # the detector's number of pmtxbins, rather than sig_cone
            else:
                if self.det_res.detector_r:
                    # Only approximate for curved detectors
                    pmt_width = self.det_res.pmt_side_length/(self.det_res.base*self.det_res.pmtxbins)
                else:
                    pmt_width = self.det_res.pmt_side_length/self.det_res.pmtxbins # From config; assume pmtxbins=pmtybins
                # To get angular resolution: divide 2*pi solid angle by the total number of PMTs behind each lens
                # Then set variance equal to that for a 2D circular top hat profile of the same angular width
                # Uses the small angle approximation
                sig_th = sig_cone#2/(np.sqrt(np.pi)*lens_dia/pmt_width)
                sigmas.fill(sig_th)
                # To get starting position resolution, set variance equal to that for a 2D circular 
                # top hat profile of the same area.
                # Then break up into two perpendicular components: sig_u^2+sig_v^2=sig_r^2, sig_u=sig_v
                sig_u = lens_dia/4#lens_dia/(np.sqrt(np.pi)*lens_dia/pmt_width)
                hit_face_array, _, _ = self.det_res.pmt_bin_to_tuple(event_pmt_bin_array, self.det_res.pmtxbins, self.det_res.pmtybins)
                print hit_face_array
                quit() 
                face_dir_array = self.det_res.direction[hit_face_array] # (n, 3)
                u_dir_array = normalize(np.cross(face_dir_array,np.array([0,0,1]))) # Check dimensions!
                v_dir_array = normalize(np.cross(face_dir_array,u_dir_array))
                u_noise = np.random.normal(scale=sig_u,size=np.shape(end_direction_array[0,:]))
                v_noise = np.random.normal(scale=sig_u,size=np.shape(end_direction_array[0,:]))
                hit_noise = u_noise*u_dir_array.T + v_noise*v_dir_array.T # (3, n)
                hit_pos = event_pmt_pos_array + hit_noise
                if debug:
                    print "sig_u: " + str(sig_u)
                    print "sig_th: " + str(sig_th)
                    print "pmt_width: " + str(pmt_width)
                    print "inscribed radius: " + str(self.det_res.inscribed_radius)

            logger.info('Shapes: %s %s %s' % (np.shape(ending_photons), np.shape(hit_pos), np.shape(end_direction_array)))
            ang_noise = np.zeros(np.shape(end_direction_array))
            hit_pos = ending_photons.T
            # Really looks like something  is biasing the rays????  by 250mm in each direction??  (That's a very rough impression!)
            #ang_noise = np.random.normal(scale=sig_th,size=np.shape(end_direction_array))#np.zeros(np.shape(end_direction_array))#
            means = normalize((-end_direction_array+ang_noise).T).T
            tracks = Tracks(hit_pos, means, sigmas, qe=qe)
            #self.plot_tracks(tracks)
            return tracks

        else: # Detector is calibrated, use response to generate tracks
            tracks = Tracks(event_lens_pos_array,
                            self.det_res.means[:,event_pmt_bin_array],
                            self.det_res.sigmas[event_pmt_bin_array],
                            lens_rad = self.det_res.lens_rad,
                            lenses=lenses,
                            rings=rings,
                            pixels_in_ring=pixels,
                            qe=qe)
            #tracks = Tracks(event_pmt_pos_array, self.det_res.means[:,event_pmt_bin_array], self.det_res.sigmas[event_pmt_bin_array], lens_rad = 0.0000001)
            msk = tracks.sigmas > 0.001
            tracks.cull(np.where(msk)) # Remove tracks with zero uncertainty (not calibrated)
            if np.any(np.isnan(tracks.sigmas)):
                print "Nan tracks!! Removing."
                nan_tracks = np.where(np.isnan(tracks.sigmas))
                tracks.cull(nan_tracks)
            if debug:
                print "Tracks for calibrated PMTs: " + str(len(tracks))
            #tracks.cull(np.where(tracks.sigmas<0.2)) # Remove tracks with too large uncertainty
            #tracks.sigmas[:] = 0.054 # Temporary! Checking if setting all sigmas equal to each other helps or hurts
        if heat_map:
            return tracks, event_pmt_bin_array[msk]
        return tracks

    @staticmethod
    def get_weights(chi, chiC, Tm):
        # Returns an array of weights, according to the AVF sigmoid weighting function
        # logger.info('Chi., weight: %d' % len(chi))
        # pprint.pprint(chi)
        result = 1.0/(1.0+np.exp((chi**2-chiC**2)/(2*Tm)))
        # pprint.pprint(result)
        return result
 
    @staticmethod
    def get_track_fit_params(tracks, vtx, chiC, Tm):
        # Calculates distance, weight, and more for tracks being associated to vtx
        # Inputs:
        # tracks - Tracks object
        # vtx - Vertex object
        # chiC - "inlier threshold" in units of sigma
        # Tm - current annealing temperature
        # Returns:
        # r - closest points along the tracks to vtx
        # sig - sigma for each track, scaled by length along track to r
        # d - distances from r to vtx
        # chi - d in units of sigma
        # wt - weight for each track
        # obj - objective function (weighted sum of squared distances)
        
        # Calculate closest points along the tracks to vtx and scale sigmas by the length along the tracks
        r, sig = tracks.closest_pts_sigmas(vtx)

        d = vtx.dist(r) # distance from tracks to current vtx
        chi = d/sig # distance in units of sigma
    
        wt = EventAnalyzer.get_weights(chi, chiC, Tm) # get track weights for current vtx

        # sig = np.ones(sig.shape) # Temporary, for debugging
        # chi = d
        # wt = np.ones(wt.shape) 
	
        obj = np.sum(wt*chi**2)/np.sum(wt) # Get current value of objective function
        return r, sig, d, chi, wt, obj
 
    # Move to utilities
    def plot_tracks(self, _tracks, pts=None, highlight_pt=None, path=None, show=True, skip_interval=200):
        # Returns a 3D plot of tracks (a Tracks object), as lines extending from their 
        # PMT hit position to the inscribed diameter of the detector. 
        # If pts is not None, will also draw them (should be a (3,n) numpy array).
        # If highlight_pt exists, it will be colored differently.
        # If path exists, a path will be drawn between its points (should be shape (3,n)).

        hit_pos = _tracks.hit_pos.T[0::skip_interval].T
        means = _tracks.means.T[0::skip_interval].T
        end_pts = hit_pos + (1.5 * self.det_res.inscribed_radius * means)  # TODO: Shouldn't need to go to 1.5 here to get the tracks to cross?
        logger.info('Plotting %d tracks' % len(hit_pos[0]))

        xs = np.vstack((hit_pos[0, :], end_pts[0, :]))
        ys = np.vstack((hit_pos[1, :], end_pts[1, :]))
        zs = np.vstack((hit_pos[2, :], end_pts[2, :]))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Draw track hit positions
        ax.scatter(hit_pos[0, :], hit_pos[1, :], hit_pos[2, :], color='red')
        # Draw tracks as lines
        for ii in range(len(hit_pos[0])):
            ax.plot(xs[:,ii],ys[:,ii],zs[:,ii],color='red')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #plt.title(plot_title)

        # Draw pts
        if pts is not None:
            ax.scatter(pts[0,:],pts[1,:],pts[2,:],color='blue')
        
        # Draw highlight_pt, larger and different color
        if highlight_pt is not None:
            ax.scatter(highlight_pt[0],highlight_pt[1],highlight_pt[2],color='green',s=50)

        # Draw path between points in path
        if path is not None:
            ax.plot(path[0,:],path[1,:],path[2,:],color='blue')
            plt.title('Vertex position record')

        if show:
            plt.show()
        
        return fig
    
    @staticmethod
    def plot_weights(wt_rec, obj=None, show=True):
        host = host_subplot(111, axes_class=AA.Axes)
        plt.subplots_adjust(right=0.75)

        host.set_ylim(0, 1)

        host.set_xlabel("Iteration")
        host.set_ylabel("Track weights")
        
        host.plot(wt_rec)#, label="Track weights")

        if obj is not None:
            par1 = host.twinx()
            par1.set_ylabel("Objective function")
            p2, = par1.plot(obj, linewidth=2, color='red')
            
            par1.axis["right"].label.set_color(p2.get_color())
            #host.legend()
        
        if show:
            plt.show()
        
        return host

    def det_performance(self, event_pdf, n, startingtype='max_starting_zone', event_location=None):
    #takes the pdf for a single event and creates an array and plot showing what portion of the pdf (probability of event) lies within a sphere of radius r centered at the chosen starting position for n evenly spaced values of r.
        shape = np.shape(event_pdf)
        x, y, z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        bin_array = zip(x.ravel(), y.ravel(), z.ravel())

        #print 'Finding max starting bin...'
        if startingtype == 'max_starting_zone':
            startingbin = self.find_max_starting_bin(bin_array, event_pdf)
            startingcoord = self.det_res.bin_to_position(startingbin)
        elif startingtype == 'mode':
            startingbin = np.unravel_index(np.argmax(event_pdf), (shape[0], shape[1], shape[2]))
            startingcoord = self.det_res.bin_to_position(startingbin)
        elif startingtype == 'center_of_mass':
            startingbin = ndimage.measurements.center_of_mass(event_pdf)
            startingcoord = self.det_res.bin_to_position(startingbin)
        elif startingtype == 'actual_position':
            startingcoord = event_location
        else:
            print "You need a starting type!"
            
        coord_array = self.det_res.bin_to_position_array() 
        tree = spatial.cKDTree(coord_array)

        probabilities = np.zeros(n)
        radii = np.linspace(0, 2*self.det_res.inscribed_radius, n)
        radii[0] = 1e-5
        for i in range(n):
            radius = radii[i]
            #print 'Querying tree...'
            neighbors = tree.query_ball_point(startingcoord, radius, eps=1e-5)
            #print 'Query finished!'
            for j in range(len(neighbors)):
                bin_tuple = bin_array[neighbors[j]]
                probabilities[i] = probabilities[i] + event_pdf[bin_tuple]
        performance_array = np.reshape(np.dstack((radii, probabilities)), (n, 2))
        #print performance_array
           
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
    
    @staticmethod
    def find_max_starting_bin(bin_array, event_pdf):
        #Finds the bin that produces the largest average value among it and its immediate neighbors. Returns a tuple for the center of mass of this bin and its immediate neighbors.
        max_starting_bin = (0, 0, 0)
        max_value = 0
        for bintup in bin_array:
            center_of_mass, average_value = self.get_com(event_pdf, bin_tup, np.shape(event_pdf))
            if average_value is None:
                continue
            if average_value > max_value:
                max_value = average_value
                max_starting_bin = center_of_mass
        return tuple(max_starting_bin)
    
    @staticmethod
    def get_com(event_pdf, bintup):
        # Returns the center of mass and average value of all bins neighboring
        # bin given by indices in bin_tup, using event_pdf values
        # Returns None if values are 0

        shape = np.shape(event_pdf)
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
             return None, None
        center_of_mass = center_of_mass/value
        #taking the average of value so that bins on the edge are not penalized:
        average_value = value/number_of_bins
        return center_of_mass, average_value

    @staticmethod
    def gaussint(radii, sigma):
        return [np.sqrt(2/np.pi)/sigma**3*integrate.quad(lambda x: x**2*np.exp(-x**2/(2.0*sigma**2)), 0, radius)[0] for radius in radii]
    
    @staticmethod
    def get_angles(r0, r, L):
        #Returns an array of angles between the vectors from r0 (one 3D position) 
        #to r (many 3D positions) and the line pointing along direction L (one 3D vector)
        
        r_vec = r-r0 # reference to r0
        r_norm = np.linalg.norm(r_vec, axis=1) # Get norms
        r_proj = np.dot(r_vec,L) # Get projections onto L
        cos_r = r_proj/r_norm
        cos_r = np.clip(cos_r, -1.0, 1.0) # Restrict to valid cosine values (avoids rounding error)
        return np.arccos(cos_r) # Return angles between r_vec and L
        
    @staticmethod
    def gauss_prob(vals, sig):
        #Returns an array of Gaussian probabilities with mean=0, sigma=sig
        #Normalizes the array, so that the sum over all is 1

        prob = np.exp(-np.power(vals,2.)/(2*sig**2)) # Gaussian form
           
		# Normalize
        p_sum = np.sum(prob)
		
        if p_sum == 0:
            return prob
        if np.isnan(p_sum):
            print "Nan encountered in gauss_prob!"
            return 
        prob /= p_sum
		# Zero out tiny values (speeds things up, makes plotting cleaner)
        tol = 1e-20 # Smaller than we possibly care about

        near_zero = prob < tol
        prob[near_zero] = 0.

        return prob # Return normalized probabilities

    @staticmethod
    def gauss_nll(vals, sig):
        #Returns an array of log likelihood values, mean=0, sigma=sig
        #Not negated, to keep the optimal value as a maximum

        nll = -np.power(vals,2.)/(sig**2) # NLL model; ignoring the factor of 2

        return nll
        
    @staticmethod
    def set_style():
        # Set matplotlib style
        #mpl.rcParams['lines.linewidth'] = 2
        mpl.rcParams['lines.markersize'] = 8
        mpl.rcParams['font.size'] = 16.0
