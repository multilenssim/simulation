import numpy as np
import pprint

def _print_bins(bins, label):
    print(label + ' counts:')
    for index, count in enumerate(bins):
        if count != 0:
            if (index == 21):
                print('    Cherenkov:     ' + str(count))
            elif (index == 22):
                print('    Scintillation: ' + str(count))
            else:
                print('    ' + str(index) + ': ' + str(count))
    print


def count_processes(output, print_counts=True):
    type_bins = np.bincount(output.process_types)
    # Count subtypes
    subtype_bins = np.bincount(output.process_subtypes)
    # Magic numbers.  For subtype definitions, see:
    #    http://geant4.web.cern.ch/geant4/collaboration/working_groups/electromagnetic/
    scint_count = 0
    if (len(subtype_bins)) >= 23:
        scint_count = subtype_bins[22]
        
    cherenkov_count = 0
    if (len(subtype_bins)) >= 22:
        cherenkov_count = subtype_bins[21]

    if print_counts:
        _print_bins(type_bins, 'Process type')
        _print_bins(subtype_bins, 'Process subtype')
    return scint_count, cherenkov_count

# Note: this is very specific to the analysis in driver_geant4_only
def display_track_tree(track_tree, particle):
    track_count = len(track_tree) - 1
    if particle == 'e-' and track_count != 1:
        print
        print("e- with " +  str(track_count) + " tracks:")
        pprint.pprint(track_tree)
        print
    if particle == 'gamma' and ((track_count < 18) or (track_count > 32)):
        print
        print("gamma with " +  str(track_count) + " tracks:")
        pprint.pprint(track_tree)
        print
