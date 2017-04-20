import argparse,time
import numpy as np
import first_test_jacopo as jacopo
import matplotlib.pyplot as plt
from chroma.sample import uniform_sphere


def substract_hist(arr_dist,double_arr_dist,bn_arr,radius,s=None,sgm=None,s_sgm=None):
	if sgm != None:
		av_wgt,s_wgt = avg_hist(arr_dist,bn_arr,sgm=s_sgm)
	else:
		av_wgt,s_wgt = avg_hist(arr_dist,bn_arr)
	lgr = len(radius)
	color = plt.cm.rainbow(np.linspace(0, 1, lgr))
	for lab,arr,i,col in zip(radius,double_arr_dist,range(lgr),color):
		wgt = []
		wgt1 = []
		np_double = np.asarray(arr)
		if sgm != None:
			c_wgt = sgm[i]
		else:
			c_wgt = np.ones(len(np_double))
		for bn in bn_arr:
			wgt1.extend(np.dot([(np_double>=bn) & (np_double<(bn+bn_arr[1]))],c_wgt))
		wgt = np.asarray(wgt1)/sum(wgt1) - av_wgt
		plt.plot(bn_arr,wgt,c=col,label='respective distance %0.2f mm'%lab,ls='steps')
	if s:
		#plt.plot(bn_arr,av_wgt)
		plt.fill_between(bn_arr,s_wgt,-s_wgt,alpha=0.2)
	plt.xlabel('respective distance between tracks [mm]')
	plt.ylabel('normalized residuals (compared to SS)')
	plt.legend(ncol=4, prop={'size': 15})
	plt.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.06)	
	plt.show()
	plt.close()

def avg_hist(ss_arr_dist,bn_arr,sgm=None):
	wgt = []
	for arr,i in zip(ss_arr_dist,range(len(ss_arr_dist))):
		wgt1 = []
		np_double = np.asarray(arr)
		if sgm != None:
			c_wgt = sgm[i]
		else:
			c_wgt = np.ones(len(np_double))
		for bn in bn_arr:
			wgt1.extend(np.dot([(np_double>=bn) & (np_double<(bn+bn_arr[1]))],c_wgt))
		wgt.append(np.asarray(wgt1)/sum(wgt1))
	wgt = np.asarray(wgt)
	av_wgt = np.mean(wgt,axis=0)
	s_wgt = np.std(wgt,axis=0)
	return av_wgt,s_wgt

def overlap_hist(arr_dist,double_arr_dist,bn_arr,radius):
	plt.hist(np.asarray(arr_dist),bins=bn_arr,normed=True,label='one source',histtype='step')
	for lab,arr in zip(radius,double_arr_dist):
		plt.hist(np.asarray(arr),bins=bn_arr,normed=True,label='respective distance %0.0f mm'%lab,histtype='step')
	plt.xlabel('respective distance between tracks [mm]')
	plt.legend()
	plt.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.06)	
	plt.show()
	plt.close()

def main(sgm,ctr):
	datadir = "/home/miladmalek/TestData/"					#"/home/skravitz/TestData/"#
	fileinfo = 'cfJiani3_4'							#'configpc6-meniscus6-fl1_027-confined'#'configpc7-meniscus6-fl1_485-confined'#'configview-meniscus6-fl2_113-confined'
	in_file = datadir+'detresang-'+fileinfo+'_noreflect_100million.root'
	max_val = 5000  							#max(arr_dist+list(np.amax(double_arr_dist)))
	bin_width = 100
	n_bin = max_val/bin_width
	ss_sample = 1
	if ctr:
		loc1 = np.zeros((ss_sample,3))
	else:
		loc1 = jacopo.sph_scatter(ss_sample)
	jacopo.plot_sphere(loc1)
	bn_arr = np.linspace(bin_width,max_val,n_bin)
	if sgm != None:
		ss_arr_dist, ss_arr_sgm = jacopo.create_event(loc1, 0.01,4000,fileinfo,in_file,sgm=sgm)
		double_arr_dist,double_arr_sgm,radius = jacopo.double_event_eff_test(fileinfo,detres=in_file,detbins=10,n_repeat=1,sig_pos=0.01,n_ph_sim=4000,n_ratio=1,n_pos=1,max_rad_frac=0.2,loc1=None,sgm=sgm)
		substract_hist(ss_arr_dist,double_arr_dist,bn_arr,radius,s=ss_arr_dist,sgm=double_arr_sgm,s_sgm=ss_arr_sgm)

	else:
		ss_arr_dist = jacopo.create_event(loc1, 0.01,4000,fileinfo,in_file)
		double_arr_dist,radius = jacopo.double_event_eff_test(fileinfo,detres=in_file,detbins=10,n_repeat=1,sig_pos=0.01,n_ph_sim=4000,n_ratio=1,n_pos=1,max_rad_frac=0.2,loc1=None)
		substract_hist(ss_arr_dist,double_arr_dist,bn_arr,radius,s=ss_arr_dist)	#overlap_hist(arr_dist,double_arr_dist,bn_arr,radius)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', help='prompt -c to keep ss events in the center', action='store_const', dest='cntr', const='', default=None)
	parser.add_argument('-s', help='prompt -s to plot the histograms weighted by the error position', action='store_const', dest='sigma', const='sgm', default=None)
	args = parser.parse_args()
	sgm = args.sigma
	ctr = bool(args.cntr)
	main(sgm,ctr)
