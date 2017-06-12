from matplotlib.colors import LogNorm
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
		if reth: ks_par.append(jacopo.make_hist(bn_arr,tr_dist,err_dist))
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
	c2 = np.sum(np.square((chi2h - bkg_hist)/bkg_hist),axis=1)/(len(bkg_hist)-1)
	return c2

def plot_cl(ks_bin,c_bkg,c_sgn,x_str):
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(ks_bin, c_bkg, 'r-')
	ax2.plot(ks_bin, 1-c_sgn, 'b-')
	ax1.set_ylim([0, 1.05])
	ax2.set_ylim([0, 1.05])
	ax1.set_xlabel(x_str)
	ax1.set_ylabel('signal efficiency', color='r')
	ax2.set_ylabel('background discrimination', color='b')
	plt.axhline(0.95, xmin=0.05,xmax=0.95, color='b', linestyle='dashed', linewidth=2)
	plt.text(ks_bin[3],0.92,'95%')
	plt.title('%i mm distance for the double site' %dist)
	plt.show()

def use_avg():
	ks_par_bkg = bkg_dist_hist(sample,bn_arr,6600,sim,analyzer,sgm=sgm)
	ks_par = fixed_dist_hist(dist,sample,bn_arr,6600,sim,analyzer,sgm=sgm)
	ks_bin = np.linspace(min(ks_par_bkg),max(ks_par),50)
	ks_hist_bkg = jacopo.make_hist(ks_bin,ks_par_bkg,np.ones(sample))
	ks_hist = jacopo.make_hist(ks_bin,ks_par,np.ones(sample))
	plot_cl(ks_bin,np.cumsum(ks_hist_bkg),np.cumsum(ks_hist),'average value of the distribution')

def use_chi2():
	bkg_hist,chi2h = jacopo.band_shell_bkg(sample,bn_arr,6600,sim,analyzer,sgm=sgm)
	ks_par = fixed_dist_hist(dist,sample,bn_arr,6600,sim,analyzer,sgm=sgm,reth=True)
	c2_s = chi2(bkg_hist,chi2h)
	c2_b = chi2(bkg_hist,ks_par)
	ks_bin = np.linspace(0,max(c2_b),50)
	sng_h = jacopo.make_hist(ks_bin,c2_s,np.ones(sample))
	bkg_h = jacopo.make_hist(ks_bin,c2_b,np.ones(sample))
	plt.plot(ks_bin,sng_h)
	plt.plot(ks_bin,bkg_h)
	plt.show()
	plt.close()
	plot_cl(ks_bin,np.cumsum(sng_h),np.cumsum(bkg_h),'reduced $\chi^2$')

def outside_tracks():
	location = jacopo.sph_scatter(1,in_shell=0,out_shell=5000)
	for lg in location:
		sim_events = jacopo.create_double_source_events(np.asarray([0,0,4500]), np.asarray([0,0,4500]), 0.01, 3300, 3300)
		for ev in sim.simulate(sim_events, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
				tracks = analyzer.generate_tracks(ev)
		a,b = jacopo.track_dist(tracks.hit_pos.T,tracks.means.T,sgm=tracks.sigmas,outlier=False,dim_len=conf.half_EPD)
		c,d = jacopo.track_dist(tracks.hit_pos.T,tracks.means.T,sgm=tracks.sigmas,outlier=True,dim_len=conf.half_EPD)
		if sgm:
			b = 1./np.asarray(b)
			d = 1./np.asarray(d)
		else:
			b = np.ones(len(a))
			d = np.ones(len(c))
		plt.plot(bn_arr,jacopo.make_hist(bn_arr,a,b))
		plt.plot(bn_arr,jacopo.make_hist(bn_arr,c,d))
		plt.show()
		'''
	plt.hist2d(A, C, bins=50, range=[[0,2000],[0,12000]], norm=LogNorm())
	plt.colorbar()
	plt.show()'''

if __name__ == '__main__':
	max_val = 2000
	bin_width = 10
	n_bin = max_val/bin_width
	sample = 10
	dist = 100
	sgm = True
	outlier = False
	bn_arr = np.linspace(0,max_val,n_bin)
	sim,analyzer = jacopo.sim_setup('cfJiani3_4','/home/miladmalek/TestData/detresang-cfJiani3_4_1DVariance_100million.root')
	conf = detectorconfig.configdict['cfJiani3_4']
	use_avg()
	#use_chi2()
	#outside_tracks()



#('cfSam1_1','/home/miladmalek/TestData/detresang-cfSam1_1_1DVariance_noreflect_100million.root')
