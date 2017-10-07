import argparse, pickle, os
import detectorconfig

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('cfg',help='configuration name')
	args = parser.parse_args()
	cfg = args.cfg
	ptf = '/home/ubuntu/Development/TestData/'+cfg+'/raw_data/'
        if not os.path.exists(ptf):
                os.makedirs(ptf)
	with open(ptf+'conf.pkl', 'w') as outf:
    		pickle.dump(detectorconfig.configdict[cfg].__dict__, outf)
	print 'configuration saved'
