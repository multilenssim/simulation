


########################################################################
########################## DEPRECATED ##################################
########################################################################



from matplotlib.colors import LogNorm
#from scipy.stats import chisquare
import first_test_jacopo as jacopo
import matplotlib.pyplot as plt
import kabamland2 as kbl
import numpy as np
import time
import detectorconfig

def fixed_dist_hist(dist,sample,bn_arr,amount,sim,analyzer,sgm=False,plot=False,sigma=0.01,reth=False):
	ks_par = []
	i = 0
	locs1, locs2, rad = jacopo.fixed_dist(sample,5000,rads=dist)
	if plot:
		jacopo.plot_sphere(locs1,loc2=locs2,rads=rad)
	for lc1,lc2 in zip(locs1,locs2):
		i += 1
		sim_events = jacopo.create_double_source_events(lc1, lc2, sigma, amount/2, amount/2)
		for ev in sim.simulate(sim_events, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
			tracks = analyzer.generate_tracks(ev)
		if sgm:
			tr_dist,er_dist = jacopo.track_dist(tracks.hit_pos.T,tracks.means.T,sgm=tracks.sigmas,outlier=outlier,dim_len=conf.half_EPD)
			err_dist = 1./np.asarray(er_dist)
		else:
			tr_dist = jacopo.track_dist(tracks.hit_pos.T,tracks.means.T,sgm=sgm,outlier=outlier)
			err_dist = np.ones(len(tr_dist))
		if reth: ks_par.append(jacopo.make_hist(bn_arr,tr_dist,c_wgt=err_dist))
		else: ks_par.append(np.average(tr_dist,weights=err_dist))
		if i%50 == 0:
			print i
	return ks_par

def bkg_dist_hist(sample,bn_arr,amount,sim,analyzer,sgm=False,plot=False,sigma=0.01):
	ks_par,loc_p,loc_m = [],[],[]
	i = 0
	location = jacopo.sph_scatter(sample)
	if plot:
		jacopo.plot_sphere(location)
	for lg in location:
		i += 1
		sim_event = kbl.gaussian_sphere(lg, sigma, amount)
		for ev in sim.simulate(sim_event, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
			tracks = analyzer.generate_tracks(ev)
		if sgm:
			tr_dist,er_dist = jacopo.track_dist(tracks.hit_pos.T,tracks.means.T,sgm=tracks.sigmas,outlier=outlier,dim_len=conf.half_EPD)
			err_dist = 1./np.asarray(er_dist)
		else:
			tr_dist = jacopo.track_dist(tracks.hit_pos.T,tracks.means.T,sgm=sgm,outlier=outlier)
			err_dist = np.ones(len(tr_dist))
		ks_par.append(np.average(tr_dist,weights=err_dist))
		if i%50 == 0:
			print i
	return ks_par
	
def chi2(bkg_hist,chi2h):
	chi2h = np.asarray(chi2h)
	c2 = np.sum(np.square(chi2h - bkg_hist)/bkg_hist,axis=1)/(len(bkg_hist)-1)
	return c2

def plot_cl(ks_bin,c_bkg,c_sgn,x_str,dst):
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(ks_bin, c_bkg, 'r-')
	ax2.plot(ks_bin, 1-c_sgn, 'b-')
	ax1.set_ylim([0, 1.05])
	ax2.set_ylim([0, 1.05])
	ax1.set_xlabel(x_str)
	ax1.set_ylabel('signal efficiency', color='r')
	ax2.set_ylabel('background rejection', color='b')
	plt.axhline(0.95, xmin=0.05,xmax=0.95, color='b', linestyle='dashed', linewidth=2)
	plt.text(ks_bin[3],0.92,'95%')
	plt.title('%i mm distance for the double site' %dst)
	plt.show()

def use_avg():
	arr_cl = []
	ks_par_bkg = bkg_dist_hist(sample,bn_arr,6600,sim,analyzer,sgm=sgm)
	for dst in distance:
		ks_par = fixed_dist_hist(dst,sample,bn_arr,6600,sim,analyzer,sgm=sgm)
		ks_bin = np.linspace(min(ks_par_bkg),max(ks_par),50)
		ks_hist_bkg = jacopo.make_hist(ks_bin,ks_par_bkg)
		sig_c = np.cumsum(ks_hist_bkg)
		ks_hist = jacopo.make_hist(ks_bin,ks_par)
		bkg_c = np.cumsum(ks_hist)
		#plot_cl(ks_bin,sig_c,bkg_c,'average value of the distribution')
		arr_cl.append(find_cl(sig_c,bkg_c,0.95))
	return arr_cl

def use_chi2():
	arr_cl = []
	bkg_hist,chi2h = jacopo.band_shell_bkg(sample,bn_arr,6600,sim,analyzer,sgm=sgm,conf=conf)
	c2_s = chi2(bkg_hist,chi2h)
	for dst in distance:
		ks_par = fixed_dist_hist(dst,sample,bn_arr,6600,sim,analyzer,sgm=sgm,reth=True)
		c2_b = chi2(bkg_hist,ks_par)
		ks_bin = np.linspace(0,max(c2_b),50)
		sng_h = jacopo.make_hist(ks_bin,c2_s)
		bkg_h = jacopo.make_hist(ks_bin,c2_b)
		sig_c = np.cumsum(sng_h)
		bkg_c = np.cumsum(bkg_h)
		#plot_cl(ks_bin,np.cumsum(sng_h),np.cumsum(bkg_h),'reduced $\chi^2$',dst)
		arr_cl.append(find_cl(sig_c,bkg_c,0.95))
	return arr_cl

def find_cl(ss_site,ms_site,cl):
	val = 1-cl
	idx = np.abs(ms_site - val).argmin()
	return ss_site[idx]

if __name__ == '__main__':
	max_val = 2000
	bin_width = 10
	n_bin = max_val/bin_width
	sample = 60
	dist = 100
	distance = np.linspace(100,700,6)
	sgm = True
	outlier = False
	bn_arr = np.linspace(0,max_val,n_bin)
	cfg = 'cfSam1_1'
	sim,analyzer = jacopo.sim_setup(cfg,'/home/miladmalek/TestData/detresang-cfSam1_1_1DVariance_100million.root')
	conf = detectorconfig.configdict[cfg]
	avg, chisq = [], []
	for i in range(5):
		avg.append(use_avg())
		chisq.append(use_chi2())
		print i
	plt.errorbar(distance,np.mean(avg,axis=0),yerr=np.std(avg,axis=0),fmt='o',label='weighted average')
	plt.errorbar(distance,np.mean(chisq,axis=0),yerr=np.std(chisq,axis=0),fmt='o',label='$\chi^2$')
	plt.xlabel('respective distance [mm]')
	plt.ylabel('signal efficiency at 95% background rejection')
	plt.title(cfg+' events seeded in 1m radius sph.')
	plt.legend(loc='upper left')
	plt.xlim(80,720)
	plt.ylim(-0.1,1.1)
	plt.show()



#'cfSam1_1'
#'/home/miladmalek/TestData/detresang-cfSam1_1_1DVariance_noreflect_100million.root'
#'cfJiani3_4'
#/home/miladmalek/TestData/detresang-cfJiani3_4_1DVariance_100million.root
