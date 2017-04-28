import argparse
import numpy as np
import first_test_jacopo as jacopo
import matplotlib.pyplot as plt


def main(sgm):
	max_val = 5000  							#max(arr_dist+list(np.amax(double_arr_dist)))
	bin_width = 100
	n_bin = max_val/bin_width
	sample = 50
	r_dist = 4
	bn_arr = np.linspace(bin_width,max_val,n_bin)
	sim,analyzer = jacopo.sim_setup('cfJiani3_2','/home/miladmalek/TestData/detresang-cfJiani3_2_1DVariance_noreflect_100million.root')
	bkg = jacopo.band_shell_bkg(sample,bn_arr,4000,sim,analyzer,4000,5000,sgm=sgm)
	sgnl,sigma_sgnl,dists = jacopo.band_shell_sgn(r_dist,sample,bn_arr,4000,sim,analyzer,sgm=sgm)
	jacopo.substract_hist(bn_arr,bkg,sgnl,sigma_sgnl,dists)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', help='prompt -s to plot the histograms weighted by the error position', action='store_const', dest='sigma', const='sgm', default=None)
	args = parser.parse_args()
	sgm = bool(args.sigma)
	main(sgm)
