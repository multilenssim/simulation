#!/usr/bin/python

import pprint
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('pickle_file', help='Pickle file name')
args = parser.parse_args()

with open(args.pickle_file, 'rb') as f:
    # The protocol version used is detected automatically, so we do not have to specify it.
    data = pickle.load(f)
    np.set_printoptions()  # threshold=np.inf)  # In order to get the full arrays
    if data is not None:
        try:
            if 'acquisition_parameters' in data:    # Special case for DM Radio
                pprint.pprint(data.acquisition_parameters.__dict__)
        except TypeError:
            pass     # object is not iterable, just ignore this if fails as its only applicable to DMR

        # Special cases for chroma
        try:
            print('Detector parameters: ' + str(data.g4_detector_parameters.__dict__))
            print('Material: ' + str(data.detector_material.__dict__))
            # if 'g4_detector_parameters' in data:    # Never true - not sure why
        except AttributeError:  # TypeError: # not sure if this will be the right exception
            pass     # object is not iterable, just ignore this if fails

        try:
            pprint.pprint(data.__dict__)
        except AttributeError:  # We had this as TypeError - but that may have been speculative for DMR
            print('Not a dict')
            pprint.pprint(data)
    else:
        print("No data in file: " + args.pickle_file)
