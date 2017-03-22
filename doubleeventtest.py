if __name__ == '__main__':
    from DetectorResponse import DetectorResponse
    from DetectorResponsePDF import DetectorResponsePDF
    from DetectorResponseGaussAngle import DetectorResponseGaussAngle
    from ShortIO.root_short import ShortRootReader
    from EventAnalyzer import EventAnalyzer
    #from doubleeventtest import 
    import kabamland2 as kbl
    import lensmaterials as lm
    from chroma.sample import uniform_sphere
    from chroma.detector import Detector
    from chroma.sim import Simulation
    from chroma.loader import load_bvh
    from chroma.event import Photons
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import rcParams
    from matplotlib import cm
    
    datadir = "/home/miladmalek/TestData/"#"/home/skravitz/TestData/"#

    def double_event_eff_test(config, detres=None, detbins=10, n_repeat=10, sig_pos=0.01, n_ph_sim=300, n_ratio=10, n_pos=10, max_rad_frac=1.0, loc1=(0,0,0), sig_cone=0.01, lens_dia=None, n_ph=0, min_tracks=0.05, chiC=3., temps=[256, 0.25], tol=1.0, debug=False):
        # Produces a plot of reconstruction efficiency for double source events (# of events w/ 
        # 2 reconstructed # vtcs/# of events) as a function of event separation and ratio of source photons.
        # Creates a simulation of the given config, etc. (actual lenses set in kabamland2.py)
        # Simulates events with total number of photons given by n_ph_sim, split into two sources
        # One source is set at loc1, while the other is varied in radial distance from loc1 
        # (up to max_rad_frac*inscribed radius in n_pos steps)
        # The size of each photon source is given by sig_pos
        # The ratio of photons from each source is also varied from 0.5 to 0.99 (n_ratio steps)
        # Each (pos, ratio) pair is repeated n_repeat times
        # The event is then analyzed using the AVF algorithm to reconstruct vertex positions, and
        # the average reconstruction efficiency is recorded
        # Uses a DetectorResponse given by detres (or idealized resolution model, if detres=None)
        # AVF params (including n_ph, which can be used to fix the number of detected photons) also included
 
        kabamland = Detector(lm.ls)
        kbl.build_kabamland(kabamland, config)
        kabamland.flatten()
        kabamland.bvh = load_bvh(kabamland)
        #view(kabamland)
        #quit()

        sim = Simulation(kabamland)
        print "Simulation started."

        if detres is None:
            det_res = DetectorResponseGaussAngle(config, detbins, detbins, detbins)
        else:
            det_res = DetectorResponseGaussAngle(config, detbins, detbins, detbins, infile=(datadir+detres))
        analyzer = EventAnalyzer(det_res)

        # make list of radii and energy ratios
        # Keep events inside detector
        max_rad = min(max_rad_frac*det_res.inscribed_radius, det_res.inscribed_radius-np.linalg.norm(np.array(loc1)))
        rads = [max_rad*float(ii+1)/n_pos for ii in range(n_pos)]
        ratios = [0.5+0.49*float(ii)/max(n_ratio-1,1) for ii in range(n_ratio)]
        rad_plot = np.tile(rads/det_res.inscribed_radius, (n_ratio,1))
        ratio_plot = np.tile(ratios, (n_pos,1)).T

        effs = np.zeros((n_ratio, n_pos))
        avg_errs = np.zeros((n_ratio, n_pos))
        #avg_disps = np.zeros((n_ratio, n_pos))
        for ix, ratio in enumerate(ratios):
            amount1 = n_ph_sim*ratio
            amount2 = n_ph_sim*(1.0-ratio)
            for iy, rad in enumerate(rads):
                print "Ratio: " + str(ratio) + ", radius: " + str(rad)
                locs1 = [loc1]*n_repeat # list
                locs2 = uniform_sphere(n_repeat)*rad+loc1 #np array

                sim_events = create_double_source_events(locs1, locs2, sig_pos, amount1, amount2)

                times = []
                vtx_disp = []
                vtx_err = []
                vtx_unc = []
                n_vtcs = []
                for ind, ev in enumerate(sim.simulate(sim_events, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100)):
            
                    #print "Iteration " + str(ind+1) + " of " + str(n_repeat)
                    t0 = time.time()
                    # Do AVF event reconstruction
                    vtcs = analyzer.analyze_one_event_AVF(ev, sig_cone, n_ph, min_tracks, chiC, temps, tol, debug, lens_dia)
                    t1 = time.time()
                    # Check performance: speed, dist from recon vertex to event pos for each, uncertainty for each
                    doWeights = True # Weight vertices by n_ph
                    if vtcs: # Append results unless no vertices were found
                        times.append(t1-t0)
                        n_vtcs.append(len(vtcs))
                        vtx_unc.append(np.mean([vtx.err for vtx in vtcs])) # weight by vtx.n_ph?
                        event_pos = [locs1[ind], locs2[ind]]
                        if event_pos is not None:
                            min_errs = []
                            weights = []
                            for ii, vtx in enumerate(vtcs):
                                print "vtx "+str(ii)+" radius: ", np.linalg.norm(vtx.pos) 
                                if np.linalg.norm(vtx.pos) > det_res.inscribed_radius: # Skip vertices outside detector
                                    break 
                                errs = [vtx.pos-ev_pos for ev_pos in event_pos] # Get distances to true source locs
                                min_ind = np.argmin([np.linalg.norm(err) for err in errs])
                                if doWeights:
                                    min_errs.append(errs[min_ind]*vtx.n_ph)
                                    weights.append(vtx.n_ph) 
                                else:
									min_errs.append(errs[min_ind])
									weights.append(1.0)
                                #break #To use only the first vtx found
                            #vtx_err = np.linalg.norm(min_errs)
							#print "n vertices: ", len(vtcs)
                            vtx_err_dist = np.linalg.norm(min_errs, axis=1)
                            vtx_avg_dist = np.sum(vtx_err_dist)/np.sum(weights)
                            vtx_err_disp = np.sum(min_errs,axis=0)/np.sum(weights)
							#print "average displacement to true event: ", vtx_err_disp
							#print "average distance to true event: ", vtx_avg_dist
							
                            vtx_disp.append(vtx_err_disp)
                            vtx_err.append(vtx_avg_dist)
                effs[ix, iy] = np.mean([1.0*(int(nii)==2) for nii in n_vtcs])
                avg_errs[ix, iy] = np.mean(vtx_err)
                print "avg displacement: ", np.mean(vtx_disp,axis=0)
                print "avg distance: ", np.mean(vtx_err)
                print "avg n_vtcs: " + str(np.mean(n_vtcs))
                #print "avg time: " + str(np.mean(times))

        #Lists of efficiencies, errors, displacements generated
        # print rad_plot
        # print ratio_plot
        print "Two-Vertex Reconstruction Efficiencies: ", effs
        print "Average Photon-Weighted Euclidean Distance to True Vertex: ", avg_errs
        plot_eff_contours(rad_plot, ratio_plot, effs)

    def create_double_source_events(locs1, locs2, sigma, amount1, amount2):
        # produces a list of Photons objects, each with two different photon sources
        # locs1 and locs2 are lists of locations (tuples)
        # other parameters are single numbers
        events = []
        n_loc = min(len(locs1),len(locs2))
        for ind in range(n_loc):
            loc1 = locs1[ind]
            loc2 = locs2[ind]
            event1 = kbl.gaussian_sphere(loc1, sigma, amount1)
            event2 = kbl.gaussian_sphere(loc2, sigma, amount2)
            event = event1 + event2 # Just add the list of photons from the two sources into a single event
            events.append(event)
        return events

    def plot_eff_contours(X, Y, Z):
        # Plot surface of efficiency (Z) versus radius (X) and photon ratio (Y)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.3)
        zmin = -0.2
        cset = ax.contourf(X, Y, Z, zdir='z', offset=zmin, cmap=cm.coolwarm)
        # Can draw 
        #cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
        #cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

        ax.set_xlabel('Separation/Detector Radius')
        #ax.set_xlim(-40, 40)
        ax.set_ylabel(r'E$_1$/(E$_1$+E$_2$)')
        ax.set_ylim(0.5, 1.0)
        ax.set_zlabel('Two-Vertex Reconstruction Efficiency')
        ax.set_zlim(zmin, 1.0)
        plt.colorbar(cset)

        plt.show()

    def double_event_recon(config, detres=None, detbins=10, sig_pos=0.01, n_ph_sim=300, ratio=0.5, rad_frac=0.5, loc1=(0,0,0), sig_cone=0.01, lens_dia=None, n_ph=0, min_tracks=0.05, chiC=3., temps=[256, 0.25], tol=1.0, debug=False):
        # Reconstructs one double-source event - mostly for testing purposes
        # Creates a simulation of the given config, etc. (actual lenses set in kabamland2.py)
        # Simulates events with total number of photons given by n_ph_sim, split into two sources
        # One source is set at loc1, while the other is varied in radial distance from loc1 
        # (up to max_rad_frac*inscribed radius in n_pos steps)
        # The size of each photon source is given by sig_pos
        # The ratio of photons from each source is also varied from 0.5 to 0.99 (n_ratio steps)
        # The event is then analyzed using the AVF algorithm to reconstruct vertex positions
        # Uses a DetectorResponse given by detres (or idealized resolution model, if detres=None)
        # AVF params (including n_ph, which can be used to fix the number of detected photons) also included
        kabamland = Detector(lm.ls)
        kbl.build_kabamland(kabamland, config)
        kabamland.flatten()
        kabamland.bvh = load_bvh(kabamland)
        #view(kabamland)
        #quit()

        sim = Simulation(kabamland)
        print "Simulation started."

        if detres is None:
            det_res = DetectorResponseGaussAngle(config, detbins, detbins, detbins)
        else:
            det_res = DetectorResponseGaussAngle(config, detbins, detbins, detbins, infile=(datadir+detres))
        analyzer = EventAnalyzer(det_res)

        
        loc2 = tuple((uniform_sphere(1)*rad_frac*det_res.inscribed_radius).flatten()+loc1)
        amount1 = n_ph_sim*ratio
        amount2 = n_ph_sim*(1-ratio)

        # Create and simulate event
        photons = create_double_source_events([loc1], [loc2], sig_pos, amount1, amount2)
        for ev in sim.simulate(photons, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
            # Do AVF step to reconstruct vertices
            vtcs = analyzer.analyze_one_event_AVF(ev, sig_cone, n_ph, min_tracks, chiC, temps, tol, debug, lens_dia)
            
            # Check performance: speed, dist from recon vertex to event pos for each, uncertainty for each
            doWeights = True # Weight vertices by n_ph
            if vtcs: # Append results unless no vertices were found
                n_vtcs = len(vtcs)
                event_pos = [loc1, loc2]
                if event_pos is not None:
                    min_errs = []
                    weights = []
                    for vtx in vtcs:
						if np.linalg.norm(vtx.pos) > det_res.inscribed_radius: # Skip vertices outside detector
							break 
						errs = [vtx.pos-ev_pos for ev_pos in event_pos] # Get distances to true source locs
						min_ind = np.argmin([np.linalg.norm(err) for err in errs])
						if doWeights:
							min_errs.append(errs[min_ind]*vtx.n_ph)
							weights.append(vtx.n_ph) 
						else:
							min_errs.append(errs[min_ind])
							weights.append(1.0)
						#break #To use only the first vtx found
                    print "n vertices: ", len(vtcs)
                    vtx_err_dist = np.linalg.norm(min_errs, axis=1)
                    vtx_avg_dist = np.sum(vtx_err_dist)/np.sum(weights)
                    vtx_err_disp = np.sum(min_errs,axis=0)/np.sum(weights)
                    print "average displacement to true event: ", vtx_err_disp
                    print "average distance to true event: ", vtx_avg_dist

    def set_style():
        # Set matplotlib style
        #rcParams['lines.linewidth'] = 2
        rcParams['lines.markersize'] = 8
        rcParams['font.size'] = 16.0
        rcParams['figure.figsize'] = (12, 9)
    

    set_style()

    fileinfo = 'cfJiani3_4'#'configpc6-meniscus6-fl1_027-confined'#'configpc7-meniscus6-fl1_485-confined'#'configview-meniscus6-fl2_113-confined'

    double_event_eff_test('cfJiani3_4', detres='detresang-'+fileinfo+'_noreflect_100million.root', detbins=10, n_repeat=10, sig_pos=0.01, n_ph_sim=4000, n_ratio=10, n_pos=5, max_rad_frac=0.2, loc1=(0,0,0), sig_cone=0.01, lens_dia=None, n_ph=0, min_tracks=0.05, chiC=2., temps=[256, 0.25], tol=1.0, debug=False)


    #double_event_eff_test('configpc6', detres=None, detbins=10, n_repeat=2, sig_pos=0.1, n_ph_sim=300, n_ratio=2, n_pos=2, max_rad_frac=1.0, loc1=(0,0,0), sig_cone=0.01, lens_dia=None, n_ph=0, min_tracks=0.05, chiC=3., temps=[256, 0.25], tol=1.0, debug=True)

    #double_event_recon('cfJiani3_2', detres='detresang-'+fileinfo+'_noreflect_100million.root', detbins=10, sig_pos=0.01, n_ph_sim=4000, ratio=0.9, rad_frac=0.6, loc1=(0,0,0), sig_cone=0.01, lens_dia=None, n_ph=0, min_tracks=0.05, chiC=1.5, temps=[256, 0.25], tol=1.0, debug=True)

    
