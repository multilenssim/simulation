from DetectorResponse import DetectorResponse
from DetectorResponsePDF import DetectorResponsePDF
from DetectorResponseGaussAngle import DetectorResponseGaussAngle
#from DetectorResponseGAKW import DetectorResponseGAKW
from ShortIO.root_short import ShortRootReader
from EventAnalyzer import EventAnalyzer
import detectorconfig

import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

from logger_lfd import logger
import paths

# This is deprecated - see calibrate.py
# And don't use pickle!!
#@deprecated
def create_detres(config, simname, detresname, detxbins=10, detybins=10, detzbins=10, method="PDF", nevents=-1, datadir="", fast_calibration=False):
	#saves a detector response list of pdfs- 1 for each pixel- given a simulation file of photons emitted isotropically throughout the detector.
	logger.info('Calibrating for: ' + datadir + simname + ' --- ' + method)
	if method=="PDF":
		smalltest = DetectorResponsePDF(config, detxbins, detybins, detzbins)
	elif method=="GaussAngle":
		smalltest = DetectorResponseGaussAngle(config, detxbins, detybins, detzbins)
	else:
		print "Warning: using generic DetectorResponse base class."
		smalltest = DetectorResponse(config)

	smalltest.calibrate(datadir + simname, datadir, nevents, fast_calibration=fast_calibration)
	logger.info("=== Detector analysis calibration complete.  Writing calibration file")
	#smalltest.calibrate_old(datadir + simname, nevents)
	# print smalltest.means[68260]
	# print smalltest.sigmas[68260]
	with open(datadir + detresname + '.pickle', 'wb') as outf:
		pickle.dump(smalltest, outf)
	smalltest.write_to_ROOT(datadir + detresname)
	
def reconstruct_event_PDF(config, detres, event, event_pos=None, datadir=""):
	#runs the analyze_event function for a given configuration and event file
	nexttest = DetectorResponsePDF(config, infile=(datadir+detres))
	print "DetectorResponse created."
	#nexttest.read_from_ROOT(datadir + detres) #Read in during the initialization
	analyzer = EventAnalyzer(nexttest)
	analyzer.analyze_event_PDF(datadir + event, event_pos)

def plot_light_cone(config, detres, pmtchoice):
	conetest = DetectorResponsePDF(config, infile=(datadir+detres))
	#conetest.read_from_ROOT(datadir + detres) #Read in during the initialization
	conetest.plot_pdf(conetest.pdfs[pmtchoice], 'PDF of pmt ' + str(pmtchoice))

def create_anglesres(config, simname, nolens=False, rmax_frac=1.0, datadir=""):
	#creates a histogram of angles of light hitting detectors
	#most useful if the simulation is of "perfectres" type, i.e. no lenses to change light angles
	#or "anglestest" type, i.e. lenses are replaced by PMTs
	#Only counts light within rmax_frac*inscribed_radius of the center
	anglestest = DetectorResponse(config)
	anglestest.angles_response(config, datadir + simname, nolens=nolens, rmax_frac=rmax_frac) 

def create_perfect_res_dir_list(config, detxbins, detybins, detzbins, simname, filename, datadir=""):
	#creates and saves the detector direction list created in the build_perfect_resolution_direction_list function 
	perfectrestest = DetectorResponse(config, detxbins, detybins, detzbins)
	perfectrestest.build_perfect_resolution_direction_list(datadir + simname, datadir + filename)

def reconstruct_perfect_res_event(config, perfect_res_dir_list_file, event, event_pos=None, detbins=10, recon_mode='angle', sig_cone=0.01, n_ph=0):
	nextperfecttest = DetectorResponse(config, detbins, detbins, detbins)
	analyzer = EventAnalyzer(nextperfecttest)
	analyzer.analyse_perfect_res_event(datadir + event, perfect_res_dir_list_file, event_pos, recon_mode, sig_cone, n_ph)

def reconstruct_event_AVF(config, event, detres=None, detbins=10, event_pos=None, sig_cone=0.01, n_ph=0, min_tracks=4, chiC=3., temps=[256, 0.25], tol=1.0, debug=False, lens_dia=None, datadir=""):
	if detres is None:
		det_res = DetectorResponseGaussAngle(config, detbins, detbins, detbins)
	else:
		det_res = DetectorResponseGaussAngle(config, detbins, detbins, detbins, infile=(datadir+detres))
		# print det_res.sigmas
		# sig_test = det_res.sigmas[np.where(det_res.sigmas>0.)]
		# print max(sig_test)
		# print np.mean(sig_test)
	analyzer = EventAnalyzer(det_res)
	reader = ShortRootReader(datadir+event)
	for ev in reader:

		# Temporary
		#plot_event(ev)
		#quit()

		# End temp

		vtcs = analyzer.analyze_one_event_AVF(ev, sig_cone, n_ph, min_tracks, chiC, temps, tol, debug, lens_dia)
		# performance checks: speed, distance from recon vertex to event position for each, uncertainty for each
		if event_pos is not None:
			print "Distance to true event pos: " + str(np.linalg.norm(vtcs[0].pos-event_pos))

def get_AVF_performance(config, event, n_repeat=10, detres=None, detbins=10, event_pos=None, sig_cone=[0.01], n_ph=[0], min_tracks=[4], chiC=[3.], temps=[[256, 0.25]], tol=[1.0], debug=False, lens_dia=None, datadir=""):
	# Calls analyze_one_event_AVF multiple times for each configuration, where a configuration is
	# a set of parameters (sig_cone to tol). Uses the last entry of a parameter's list if
	# there are fewer entries for it than the current iteration number. Prints the average
	# performance across multiple runs for each configuration. 

	if detres is None:
		det_res = DetectorResponseGaussAngle(config, detbins, detbins, detbins)
	else:
		det_res = DetectorResponseGaussAngle(config, detbins, detbins, detbins, infile=(datadir+detres))
	analyzer = EventAnalyzer(det_res)
	reader = ShortRootReader(datadir+event)
	max_iter = max([len(sig_cone),len(n_ph),len(min_tracks),len(chiC),len(temps),len(tol)])
	
	for ev in reader:
		for it in range(max_iter):
			sig_cone_i = idx_check(sig_cone, it)
			n_ph_i = idx_check(n_ph, it)
			min_tracks_i = idx_check(min_tracks, it)
			chiC_i = idx_check(chiC, it)
			temps_i = idx_check(temps, it)
			tol_i = idx_check(tol, it)
			print "sig_cone: " + str(sig_cone_i)
			print "n_ph: " + str(n_ph_i)
			print "min_tracks: " + str(min_tracks_i)
			print "chiC: " + str(chiC_i)
			print "temps: " + str(temps_i)
			print "tol: " + str(tol_i)
			times = []
			vtx_disp = []
			vtx_err = []
			vtx_unc = []
			n_vtcs = []
			for ind in range(n_repeat):
				print "Iteration " + str(ind+1) + " of " + str(n_repeat)
				t0 = time.time()
				vtcs = analyzer.analyze_one_event_AVF(ev, sig_cone_i, n_ph_i, min_tracks_i, chiC_i, temps_i, tol_i, debug, lens_dia)
				t1 = time.time()
				# Check performance: speed, dist from recon vertex to event pos for each, uncertainty for each
				doWeights = True # Weight vertices by n_ph
				if vtcs: # Append results unless no vertices were found
					times.append(t1-t0)
					n_vtcs.append(len(vtcs))
					vtx_unc.append(np.mean([vtx.err for vtx in vtcs])) # weight by vtx.n_ph?
					if event_pos is not None:
						min_errs = []
						weights = []
						#print vtcs[0].n_ph
						for vtx in vtcs:
							if np.linalg.norm(vtx.pos) > det_res.inscribed_radius: # Skip vertices outside detector
								break 
							errs = [vtx.pos-ev_pos for ev_pos in event_pos]
							min_ind = np.argmin([np.linalg.norm(err) for err in errs])
							if doWeights:
								min_errs.append(errs[min_ind]*vtx.n_ph)
								weights.append(vtx.n_ph) 
							else:
								min_errs.append(errs[min_ind])
								weights.append(1.0)
							#break #To use only the first vtx found
						avg_err = np.sum(min_errs,axis=0)/np.sum(weights)
						vtx_disp.append(avg_err)
						vtx_err.append(np.linalg.norm(avg_err))
			# Print mean values instead?
			print "vtx_err: " + str(vtx_err) # weighted distance of vertices to true event position
			print "mean vtx_err: " + str(np.mean(vtx_err)) # averaged over all iterations
			print "mean vtx_disp: " + str(np.mean(vtx_disp,axis=0)) #vtx-event
			print "avg n_vtcs: " + str(np.mean(n_vtcs))
			# print "vtx_unc: " + str(vtx_unc)
			# print np.mean(vtx_unc)
			print "times: " + str(times)
			print "mean time: " + str(np.mean(times))

def idx_check(lst, ind):
	if ind >= len(lst):
		return lst[-1]
	else:
		return lst[ind]

def check_detres_sigmas(config, detres, calibration_dir=""):
	det_res = DetectorResponseGaussAngle(config, infile=(calibration_dir + detres))
	sig_test = det_res.sigmas[np.where(det_res.sigmas>0.0001)]
	print "Total PMT bins: " + str(np.shape(det_res.sigmas))
	print "Calibrated PMT bins: " + str(np.shape(sig_test))
	print "Mean sigma: " + str(np.mean(sig_test))
	print "Max sigma: " + str(max(sig_test))
	max_ang = 0.3
	max_y = 60
	plt.hist(sig_test,100,normed=True,range=(0.,max_ang))
	plt.axis([0.,max_ang, 0., max_y])
	plt.xlabel('Incident Angle Std (rad)')
	plt.ylabel('PMTs/Unit Angle')
	plt.show()

	# Find PMTs with large sigma
	ind_peak2 = np.array(np.where(det_res.sigmas>0.1))[0] #np.array(np.where(np.logical_and(det_res.sigmas>0.1,det_res.sigmas>0.1)))[0]
	print 'PMTs with high angular uncertainty: ',len(ind_peak2)
	single_face = True # Draw only a single face of the icosahedron
	if single_face:
		ind_peak2 = ind_peak2[np.where(ind_peak2<det_res.n_pmts_per_surf*det_res.n_lens_sys)]
	pos_peak2 = det_res.pmt_bin_to_position(ind_peak2)

	# Get first PMT location for each curved surface (center of curved surface)
	center_pmt_bins = np.linspace(0, det_res.npmt_bins, det_res.n_lens_sys*20, endpoint=False, dtype=np.int)
	if single_face:
		center_pmt_bins = center_pmt_bins[np.where(center_pmt_bins<det_res.n_pmts_per_surf*det_res.n_lens_sys)]
	#print len(center_pmt_bins)
	center_pmt_pos = det_res.pmt_bin_to_position(center_pmt_bins)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(pos_peak2[:,0], pos_peak2[:,1],pos_peak2[:,2],color='blue')
	ax.scatter(center_pmt_pos[:,0],center_pmt_pos[:,1],center_pmt_pos[:,2],color='red')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title('PMTs with high angular uncertainty')
	plt.show()

def compare_sigmas(config1, detres1, config2, detres2, datadir=""):
	det_res1 = DetectorResponseGaussAngle(config1, infile=(datadir+detres1))
	det_res2 = DetectorResponseGaussAngle(config2, infile=(datadir+detres2))
	npmt = det_res1.npmt_bins
	if npmt != det_res2.npmt_bins:
		print "Can't compare the detector responses - different numbers of PMTs!"
		return
	calibrated_pmts = np.where(np.logical_and(det_res1.sigmas > 0.001, det_res2.sigmas > 0.001))[0]
	#print det_res1.means[:,calibrated_pmts], np.shape(det_res1.means[:,calibrated_pmts])
	cos_ang = np.einsum('ij,ij->j',det_res1.means[:,calibrated_pmts], det_res2.means[:,calibrated_pmts])
	ang = np.rad2deg(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
	#mean_diff = det_res1.means[:,calibrated_pmts] - det_res2.means[:,calibrated_pmts]
	max_ang = 15
	max_y = 0.6
	plt.hist(ang,100,normed=True,range=(0.,max_ang))
	plt.axis([0.,max_ang, 0., max_y])
	plt.xlabel('Relative angle (deg)')
	plt.ylabel('PMTs/Unit Angle')
	plt.show()
	

def testing(config):
	#use this to test various functions in DetectorResponse
	test = DetectorResponse(config)
	return test.pmt_bin_to_position(9200)

def plot_events_from_file(filename, num_events=1, num_ph=-1):
	reader = ShortRootReader(datadir+filename)
	for ii, ev in enumerate(reader):
		if ii >= num_events:
			break
		print "Event " + str(ii)
		plot_event(ev, num_ph)

def plot_event(ev, num_ph=-1):
	detected = (ev.photons_end.flags & (0x1 <<2)).astype(bool) 
	
	nohit = (ev.photons_end.flags & (0x1 << 0)).astype(bool)
	abort = (ev.photons_end.flags & (0x1 << 31)).astype(bool)
	surface_absorb = (ev.photons_end.flags & (0x1 << 3)).astype(bool)
	bulk_absorb = (ev.photons_end.flags & (0x1 << 1)).astype(bool)
	refl_diff = (ev.photons_end.flags & (0x1 << 5)).astype(bool)
	refl_spec = (ev.photons_end.flags & (0x1 << 6)).astype(bool)
	scatter = (ev.photons_end.flags & (0x1 << 4)).astype(bool)
	surf_reemit = (ev.photons_end.flags & (0x1 << 7)).astype(bool)
	surf_trans = (ev.photons_end.flags & (0x1 << 8)).astype(bool)
	bulk_reemit = (ev.photons_end.flags & (0x1 << 9)).astype(bool)
	detect_refl = refl_spec & detected
	print 'detected ' + str(np.sum(detected))
	print 'nohit ' + str(np.sum(nohit))
	print 'surface absorb '+ str(np.sum(surface_absorb))
	print 'reflect diffuse '+ str(np.sum(refl_diff))
	print 'reflect specular '+ str(np.sum(refl_spec))
	print 'bulk absorb ' + str(np.sum(bulk_absorb))
	print 'aborted ' + str(np.sum(abort))
	print 'sum of photon detect/absorb/no hit/abort categories ' + str(np.sum(detected)+np.sum(nohit)+np.sum(surface_absorb)+np.sum(bulk_absorb)+np.sum(abort))
	print np.sum(scatter), np.sum(surf_reemit), np.sum(surf_trans), np.sum(bulk_reemit)
	print 'others ' + str(np.sum(scatter)+np.sum(surf_reemit)+np.sum(surf_trans)+np.sum(bulk_reemit))
	print 'reflected and detected '+str(np.sum(detect_refl))
	print 'total photons ' + str(len(ev.photons_end))

	anom = np.logical_not(detected) & np.logical_not(nohit) & np.logical_not(surface_absorb) & np.logical_not(bulk_absorb) & np.logical_not(abort) # What's left?? Looks like reflected photons that weren't aborted, detected, etc.
	#print ev.photons_end.flags[anom]

	beginning_photons = (ev.photons_beg.pos[detected])#[:999]
	ending_photons = (ev.photons_end.pos[detected])#[:999]
	beg_abs = (ev.photons_beg.pos[surface_absorb])#[:999]
	end_abs = (ev.photons_end.pos[surface_absorb])#[:999]

	if num_ph > 0 and num_ph <= len(beginning_photons):
		beginning_photons = beginning_photons[:num_ph]
		ending_photons = ending_photons[:num_ph]
		beg_abs = beg_abs[:num_ph]
		end_abs = end_abs[:num_ph]

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	# Draw track hit positions
	ax.scatter(beginning_photons[:,0],beginning_photons[:,1],beginning_photons[:,2],color='red')
	ax.scatter(ending_photons[:,0],ending_photons[:,1],ending_photons[:,2],color='blue')
	# Draw tracks as lines
	for ii in range(len(beginning_photons[:,0])):
		ax.plot([beginning_photons[ii,0].tolist(),ending_photons[ii,0].tolist()], [beginning_photons[ii,1].tolist(),ending_photons[ii,1].tolist()], [beginning_photons[ii,2].tolist(),ending_photons[ii,2].tolist()], color='green')
	
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	plt.title('Detected photon end pos')

	draw_absorb = False # Set to true to draw positions of absorbed photons as well
	if draw_absorb:
		fig2 = plt.figure()
		ax2 = fig2.add_subplot(111, projection='3d')
		# Draw track hit positions
		#ax2.scatter(beg_abs[:,0],beg_abs[:,1],beg_abs[:,2],color='red')
		ax2.scatter(end_abs[:,0],end_abs[:,1],end_abs[:,2],color='blue')
		ax2.set_xlabel('X')
		ax2.set_ylabel('Y')
		ax2.set_zlabel('Z')
		plt.title('Absorbed photon end pos')

	plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_name', help='detector configuration', nargs='?', default='cfSam1_K200_10')
    _args = parser.parse_args()

    config_name = _args.config_name
    config = detectorconfig.get_detector_config(config_name)

    # This is really a global
    cal_file = paths.get_calibration_file_name(config_name)
    datadir = paths.get_data_file_path_no_raw(config_name)  # "/home/miladmalek/TestData/"#"/home/skravitz/TestData/"#

    #fileinfo = 'cfJiani3_3'#'configpc6-meniscus6-fl1_027-confined'#'configpc7-pcrad09dia-fl2-confined'#'configview-meniscus6-fl2_113-confined'

    check_detres_sigmas(config, cal_file)

    reconstruct_perfect_res_event(config, cal_file, 'event-configpc7-f_l-1-(3-3-3)-01-100000.root', event_pos=(3,3,3), detbins=50, recon_mode='cos', sig_cone=0.01)

    #plot_events_from_file('sim-'+fileinfo+'_100million.root', num_events=1, num_ph=100)

    #print testing('configpc7')

    #create_perfect_res_dir_list('configpc7', 50, 50, 50, 'sim-configpc7-f_l-1-perfectres-billion.root', 'detres-configpc7-f_l-1-perfectres-halfbillion-50**3detbins.root')

    # Uses angle or cos method of reconstruction; omits optional parameters
    #reconstruct_perfect_res_event('configpc7', 'detres-configpc7-f_l-1-perfectres-halfbillion.root', 'event-configpc7-f_l-1-(3-3-3)-01-100000.root', event_pos=(3,3,3))

    # Uses cone method of reconstruction
    # reconstruct_perfect_res_event('configpc7', 'detres-configpc7-f_l-1-perfectres-halfbillion.root', 'event-configpc7-f_l-1-(0-0-0)-(3-3-3)-01-100.root', event_pos=(3,3,3), recon_mode='cone', sig_cone=0.01, n_ph=100)
    #reconstruct_perfect_res_event('configpc7', 'detres-configpc7-f_l-1-perfectres-halfbillion.root', 'event-configpc7-f_l-1-(3-3-3)-01-100000.root', event_pos=(3,3,3), detbins=50, recon_mode='cone', sig_cone=0.01, n_ph=10)

    #create_detres('configpc2', 'sim-configpc2-million.root', 'detresang-configpc2-million.root', fl=0.12830005982, method="GaussAngle")

    #create_detres('configpc6', 'sim-'+fileinfo+'_100million.root', 'detresang-'+fileinfo+'_100million.root', fl=1.027, method="GaussAngle", nevents=1000)

    #check_detres_sigmas('configpc7', 'detresang-'+fileinfo+'_100million.root')

    #reconstruct_perfect_res_event('configpc7', 'detres-configpc7-f_l-1-perfectres-halfbillion.root', 'event-configpc7-f_l-1-(3-3-3)-01-100000.root', event_pos=(3,3,3), detbins=50, recon_mode='cos', sig_cone=0.01)
    
    #reconstruct_event_AVF('configpc7', 'event-configpc7-f_l-1-(3-3-3)-01-100000.root', detres='detresang-configpc4-billion.root', event_pos=(3,3,3), n_ph=100, chiC=2., temps=[256, 65, 16, 4, 1], debug=True)

    #reconstruct_event_AVF('configview', 'event-configview-meniscus6-fl2_113-confined-(3-3-3)-01-100000.root', detres='detresang-configview-meniscus6-fl2_113-confined_100million.root', fl=2.113, event_pos=(3,3,3), n_ph=50, chiC=2., temps=[256, 0.25], debug=True)

    #reconstruct_event_AVF('cfJiani3_2', 'event-cfJiani3_2-(0-0-0)-100000.root', detres='detresang-cfJiani3_2_noreflect_100million.root', event_pos=(0,0,0), chiC=3., n_ph=100, debug=True)

    #create_detres('cfJiani3_3', 'sim-'+fileinfo+'_100million.root', 'detresang-'+fileinfo+'_noreflect_100million.root', method="GaussAngle", nevents=-1)
    
    #check_detres_sigmas('cfJiani3_2', 'detresang-'+fileinfo+'_noreflect_100million.root')

    #get_AVF_performance('cfJiani3_2', 'event-'+fileinfo+'-(0-0-0)-100000.root', detres='detresang-'+fileinfo+'_noreflect_100million.root', detbins=10, n_repeat=5, event_pos=[(0.,0.,0.)], n_ph=[100], min_tracks=[0.05], chiC=[3.0], temps=[[256,0.25]], debug=True)

    #get_AVF_performance('configpc7', 'event-configpc7-f_l-1-(3-3-3)-01-100000.root', detbins=10, n_repeat=50, event_pos=(3,3,3), n_ph=[10000], chiC=[4.0,10.0], sig_cone=[0.2], debug=False, lens_dia=0.96)

    #reconstruct_event_AVF('configpc6', 'event-'+fileinfo+'-(0-0-0)-01-100000.root', detres='detresang-configpc6-meniscus6-fl1_027-confined_100million.root', event_pos=(0,0,0), n_ph=100, debug=True, fl=1.027)

    #reconstruct_event_AVF('configpc7', 'event-configpc7-f_l-1-(3-3-3)-01-100000.root', event_pos=(3,3,3), n_ph=10, sig_cone=0.2, debug=True, lens_dia=0.481)

    #create_anglesres('configpc6', 'sim-configpc6-billion-anglestest.root',fl=1.0)

    #create_anglesres('configpc7', 'sim-configpc7-f_l-1-perfectres-100million.root',fl=1.0, nolens=True, rmax_frac=0.75)

    #create_detres('configpc7', 'sim-configpc7-meniscus6-billion.root', 'detres-configpc7-meniscus6-billion.root')
    
    #reconstruct_event_PDF('configpc7', 'detres-configpc7-meniscus6-billion.root', 'event-configpc7-meniscus6-(3-3-3)-01-100000.root', event_pos=(3,3,3))

    #plot_light_cone('configpc3', 'detres-configpc3-billion.root', 68260)

    
