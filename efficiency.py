if __name__ == '__main__':
    from DetectorResponse import DetectorResponse
    from DetectorResponsePDF import DetectorResponsePDF
    from DetectorResponseGaussAngle import DetectorResponseGaussAngle
    from ShortIO.root_short import ShortRootReader
    from EventAnalyzer import EventAnalyzer





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
    import detectorconfig
    import math
    
    datadir = "/home/exo/"
    
    def eff_test(config, detres=None, detbins=10, n_repeat=10, sig_pos=0.01, n_ph_sim=300, n_ratio=10, n_pos=10, max_rad_frac=1.0, loc1=(0,0,0), sig_cone=0.01, lens_dia=None, n_ph=0, min_tracks=0.05, chiC=3., temps=[256, 0.25], tol=0.1, debug=False):
		###############################################
		
		
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
        
        max_rad = 6000
        #previous definition of rads 
        #rads = [max_rad*float(ii+1)/n_pos for ii in range(n_pos)]
        
        rads = radius_equal_vol(max_rad = max_rad, steps = n_pos)
        
        rad_plot = np.tile(rads/det_res.inscribed_radius, (n_ratio,1))
 
        #print "rads: ", rads
        #print "rad_plot: ", rad_plot
        
        effs = np.zeros((n_ratio, n_pos))
        avg_errs = np.zeros((n_ratio, n_pos))
        amount = n_ph_sim
        
        repetition = 30
        
        #energies = [500,1000,2000,3000,4000,6000,8000]
        #energies = [500,1000,2000,3000,4000]
        energies = [6600]
        recon = np.zeros((len(energies),repetition, n_pos, 3))
        
        for ii in range(len(energies)):	
			amount = energies[ii]
			for iy, rad in enumerate(rads):
				print "Energy: " + str(amount) + ", radius: " + str(rad)
				print "Radius step:		", iy 
				
				events = []
	
				points = np.zeros((repetition, 3))

				for x in range(repetition):
					theta = np.arccos(np.random.uniform(-1.0, 1.0))
					phi = np.random.uniform(0.0, 2*np.pi)
					points[x,0] = rad*np.sin(theta)*np.cos(phi)
					points[x,1] = rad*np.sin(theta)*np.sin(phi)
					points[x,2] = rad*np.cos(theta)
					event = kbl.gaussian_sphere(points[x,:], sig_pos, amount)
					events.append(event)
					
				times = []
				vtx_disp = []
				vtx_err = []
				vtx_unc = []
				n_vtcs = []
				
				for ind, ev in enumerate(sim.simulate(events, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100)):
					t0 = time.time()
					# Do AVF event reconstruction
					vtcs = analyzer.analyze_one_event_AVF(ev, sig_cone, n_ph, min_tracks, chiC, temps, tol, debug, lens_dia)
					t1 = time.time()
					print t1-t0, "	sec"
					# Check performance: speed, dist from recon vertex to event pos for each, uncertainty for each
					doWeights = True # Weight vertices by n_ph
					if vtcs: # Append results unless no vertices were found
						times.append(t1-t0)
						n_vtcs.append(len(vtcs))
						vtx_unc.append(np.mean([vtx.err for vtx in vtcs])) # weight by vtx.n_ph?
						event_pos = points[ind,:]
						if event_pos is not None:
							min_errs = []
							weights = []
							for vtx in vtcs:
								
								if np.linalg.norm(vtx.pos) > det_res.inscribed_radius: # Skip vertices outside detector
									break 
								errs = [vtx.pos-ev_pos for ev_pos in event_pos] # Get distances to true source locs
								#print "Distance to true event location:		", errs 
								r_recon = np.sqrt(vtx.pos[0]*vtx.pos[0]+vtx.pos[1]*vtx.pos[1]+vtx.pos[2]*vtx.pos[2])
								print "Reconstructed radius:		", r_recon
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
							#print rad, r_recon, vtx.n_ph
							print iy, ind 
							recon[ii,ind,iy,:] = [rad, r_recon, vtx.n_ph]
							#print recon[iy,ind,:]
				#effs[iy] = np.mean([1.0*(int(nii)==2) for nii in n_vtcs])
				#avg_errs[iy] = np.mean(vtx_err)
				#print "avg displacement: ", np.mean(vtx_disp,axis=0)
				#print "avg distance: ", np.mean(vtx_err)
				#print "avg n_vtcs: " + str(np.mean(n_vtcs))
				#print "avg time: " + str(np.mean(times))
        
        #print recon   
        #fig = plt.figure(figsize=(10, 10))
        #plt.scatter(recon[:,:,:,0], recon[:,:,:,1])
        #plt.xlabel("true radius [mm]")
        #plt.ylabel("reconstructed radius [mm]")
        #t = np.arange(0.,6000.,10) 
        #plt.plot(t,t, 'r--')
        #plt.xlim(0., 5500)
        #plt.ylim(0., 5500)
        #plt.show()
        
        #fig = plt.figure(figsize=(10, 10))
        #plt.xlabel("true radius [mm]")
        #plt.ylabel("reconstructed radius [mm]")
        #for ii in range(n_pos):
			#print ii 
			#plt.scatter(recon[0,0,ii,0], np.mean(recon[0,:,ii,1]),color="b")
			#plt.errorbar(recon[0,0,ii,0], np.mean(recon[0,:,ii,1]), yerr=np.std(recon[0,:,ii,1]), linestyle="None", color="k")
		
        #for ii in range(n_pos):
			#print ii 
			#plt.scatter(recon[1,0,ii,0], np.mean(recon[1,:,ii,1]),color="r")
			#plt.errorbar(recon[1,0,ii,0], np.mean(recon[1,:,ii,1]), yerr=np.std(recon[0,:,ii,1]), linestyle="None", color="k")
		
		
        #for ii in range(n_pos):
			#print ii 
			#plt.scatter(recon[2,0,ii,0], np.mean(recon[2,:,ii,1]),color="g")
			#plt.errorbar(recon[2,0,ii,0], np.mean(recon[2,:,ii,1]), yerr=np.std(recon[0,:,ii,1]), linestyle="None", color="k")
        #for ii in range(n_pos):
			#print ii 
			#plt.scatter(recon[2,0,ii,0], np.mean(recon[2,:,ii,1]),color="y")
			#plt.errorbar(recon[2,0,ii,0], np.mean(recon[2,:,ii,1]), yerr=np.std(recon[0,:,ii,1]), linestyle="None", color="k")
        #t = np.arange(0.,6000.,10) 
        #plt.plot(t,t, 'r--')
        #plt.show()
        
        
        ax1 = plt.gca()
        ax1.set_xlim([0,6100])
        ax2 = ax1.twinx()
        
        ax1.set_xlabel('true radius [mm]')
        ax1.set_ylabel('position resolution [%]', color='blue')
        ax2.set_ylabel('light collection efficiency [%]', color='red')
        
        
        for zz in range(len(energies)):
            for ii in range(n_pos):
			    ax1.scatter(recon[zz,0,ii,0],np.std(recon[zz,:,ii,1])/recon[zz,0,ii,0]*100, label = str(energies[zz]), color="blue")

        for zz in range(len(energies)):
			for ii in range(n_pos):
				ax2.scatter(recon[zz,0,ii,0],np.mean(recon[zz,:,ii,2]/n_ph_sim)*100, color="red")
				ax2.errorbar(recon[zz,0,ii,0], np.mean(recon[zz,:,ii,2]/n_ph_sim)*100, yerr=np.std(recon[zz,:,ii,2]/n_ph_sim)*100, linestyle="None", color="red")
        
        plt.show()
        
        
        
        ax1 = plt.gca()
        ax1.set_xlim([0,6100])
        ax2 = ax1.twinx()
        
        ax1.set_xlabel('true radius [mm]')
        ax1.set_ylabel('position resolution [%]', color='blue')
        ax2.set_ylabel('light collection efficiency [%]', color='red')
        
        
        for zz in range(len(energies)):
            for ii in range(n_pos):
			    ax1.scatter(recon[zz,0,ii,0],np.mean(recon[zz,:,ii,1])/recon[zz,0,ii,0]*100-1, label = str(energies[zz]), color="blue")

        for zz in range(len(energies)):
			for ii in range(n_pos):
				ax2.scatter(recon[zz,0,ii,0],np.mean(recon[zz,:,ii,2]/n_ph_sim)*100, color="red")
				ax2.errorbar(recon[zz,0,ii,0], np.mean(recon[zz,:,ii,2]/n_ph_sim)*100, yerr=np.std(recon[zz,:,ii,2]/n_ph_sim)*100, linestyle="None", color="red")
        
        plt.show()
        
        quit()
        
        fig = plt.figure(figsize=(10, 10))
        plt.xlabel("true radius [mm]")
        plt.ylabel("standard deviation of reconstructed radius [mm]")
        plt.ylim(0., 50)
        
        colors = ['b', 'r', 'g', 'y', 'm', 'c', 'k']
        for zz in range(len(energies)):
            for ii in range(n_pos):
			    plt.scatter(recon[zz,0,ii,0],np.std(recon[zz,:,ii,1]), color=colors[zz], label = str(energies[zz]))
		
        plt.savefig(datadir+'std.pdf')
        plt.show()
     
        
        fig = plt.figure(figsize=(10, 10))
        plt.xlabel("true radius [mm]")
        plt.ylabel("light collection efficiency")
        for ii in range(n_pos):
			plt.scatter(recon[0,ii,0],np.mean(recon[:,ii,2]/n_ph_sim),)
			plt.errorbar(recon[0,ii,0], np.mean(recon[:,ii,2]/n_ph_sim), yerr=np.std(recon[:,ii,2]/n_ph_sim), linestyle="None", color="k")
        plt.show()
        
        
        
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(recon[:,:,1],recon[:,:,2]/n_ph_sim)
        plt.xlabel("reconstructed radius [mm]")
        plt.ylabel("light collection efficiency")
        plt.show()
        
        #plot_eff_contours(recon[:,:,1],recon[:,:,0],recon[:,:,2]/n_ph_sim)

        
    def create_single_source_event(rad, sigma, amount1):
        # produces a list of Photons objects, each with two different photon sources
        # locs1 and locs2 are lists of locations (tuples)
        # other parameters are single numbers
		events = []
		points = np.empty((3))
		for x in range(10):
			theta = np.arccos(np.random.uniform(-1.0, 1.0))
			phi = np.random.uniform(0.0, 2*np.pi)
			points[0] = rad*np.sin(theta)*np.cos(phi)
			points[1] = rad*np.sin(theta)*np.sin(phi)
			points[2] = rad*np.cos(theta)
			event = kbl.gaussian_sphere(points, sigma, amount1)
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

        ax.set_xlabel('Reconstructed Radius')
        #ax.set_xlim(0., 5500)
        ax.set_ylabel('True Radius')
        #ax.set_ylim(0., 5500)
        ax.set_zlabel('Light Collection Efficiency')
        #ax.set_zlim(zmin, 1.0)
        plt.colorbar(cset)
        
        plt.show()
        
    def radius_equal_vol(steps = 11, max_rad = 6000):
		max_vol = pow(max_rad, 3)*4/3*math.pi
		#print "Volume: ", max_vol 
		rads = [math.pow(3.0/4.0*max_vol/steps/math.pi*ii, 1/3.0) for ii in range(steps)]
		print rads 
		return rads 
			
		

    def set_style():
        # Set matplotlib style
        #rcParams['lines.linewidth'] = 2
        rcParams['lines.markersize'] = 8
        rcParams['font.size'] = 16.0
        rcParams['figure.figsize'] = (12, 9)
        
    fileinfo = 'cfJiani3_2'
    #fileinfo = 'cfJiani3_test2'
    print "Efficiency test started"
    
    set_style()
    eff_test(fileinfo, detres='detresang-'+fileinfo+'_noreflect_100million.root', detbins=10, n_repeat=10, sig_pos=0.01, n_ph_sim=4000, n_ratio=10, n_pos=11, max_rad_frac=0.7, loc1=(0,0,0), sig_cone=0.01, lens_dia=None, n_ph=0, min_tracks=0.1, chiC=1.5, temps=[256, 0.25], tol=0.1, debug=False)
