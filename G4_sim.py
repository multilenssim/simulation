# Test program for creating scintillation (and other) photons from Geant4

from chroma.event import Vertex, Photons
from chroma.generator import g4gen

import matplotlib.pyplot as plt
import numpy as np
import sys

import Geant4		# Only needed to turn logging on
from Geant4.hepunit import *

import lensmaterials

class G4Generator:
	def generate(self, particle_name, position, direction, scintillator, generator, energy=2.):
		#from chroma.generator import g4gen
		gen = generator  # g4gen.G4Generator(scintillator)
		vertex = Vertex(particle_name, position, direction, energy)
		output = gen.generate_photons([vertex], mute=False)
		# print(vertex.particle_name + " photons from/direction/energy/count:\t" +
		#	  str(vertex.pos) + '\t' + str(vertex.dir) + '\t' + str(vertex.ke) + '\t' + str(len(output.pos)))

		'''
		print("====>>>> Unloading g4gen")
		# Not sure if have to do both of these
		# See: https://stackoverflow.com/questions/437589/how-do-i-unload-reload-a-python-module
		del sys.modules["chroma.generator.g4gen"]
		del g4gen
		# reload(g4gen)
		'''
		return output

	#def __del__(self):
		#print("========>>>>>>>> Destructing G4Generator instance")
		# super.__del__(self)  No call to super needed ???  It raises an error...

def compute_stats(data, x_distances):
	average = np.empty([len(data)], dtype='float')
	error = np.empty([len(data)], dtype='float')
	count = 0
	for x in x_distances:
		list = data[x]
		average[count] = np.average(list)
		error[count] = np.std(list)
		count += 1
	# We could glue all the arrays together and then not have to loop over each vector of runs: (currently they are in dictionaries
	'''
	the_array = []
	for x in x_distances:
		the_array.append(data[x])
	average2 = np.average(the_array, axis=1)
	'''
	return average, error


if __name__ == '__main__':
	'''
	Geant4.gApplyUICommand("/run/verbose 2")
	Geant4.gApplyUICommand("/event/verbose 2")
	Geant4.gApplyUICommand("/tracking/verbose 2")
	'''

	print("g4gen ref count: ", sys.getrefcount(g4gen))

	print("G4 state: ", Geant4.gStateManager.GetCurrentState())
	print("Random engine: ", Geant4.HepRandom.getTheEngine())
	print("Random seed: ", Geant4.HepRandom.getTheSeed())

	##### Debugging stuff pulled from other places #####
	# Various futzing around with world material
	# helium = G4Material.GetMaterial("G4_HE")
	# if helium == None:
	#    helium = gNistManager.FindOrBuildMaterial("G4_HE")  # Isotopes parameter?
	# If we didn't find it - squawk
	# self.world_material = helium
	# self.world_material = G4Material.GetMaterial("G4_AIR")
	# self.world_material = self.create_g4material(he)

	# Debugging stuff pulled out of g4gen.py
	element_table = Geant4.gElementTable        # Was: G4Element.GetElementTable() in c++
	# Somehow one of these blows stuff up...
	# element_table2 = gElementTable
	# material_table = gMaterialTable

	# print("create_g4material() number of elements: ", G4Element.GetNumberOfElements())

	import g4py.NISTmaterials

	# Add Geant4 PrintVersion()
	Geant4.print_version()     # Geant4 method
	#G4NistManager * NISTManager = G4NistManager::Instance();
	#NISTManager->ListMaterials("all");
	##########

	scintillator = lensmaterials.create_scintillaton_material()

	x_position = None
	if len(sys.argv) > 1:
		x_position = float(sys.argv[1])

	gen = g4gen.G4Generator(scintillator, orb_radius=7.)
	momentum = (1, 0, 0)

	# Wierd things happen when we make the world out of scintillator
	# Geant4 must be adding the orb material to the scintillator?
	# But even that doesn't explain everything (i.e. we get somewhat the expected material with air in the orb, but not with Galactic??
	#gen = g4gen.G4Generator("G4_Galactic", orb_radius=7., world_material=scintillator)
	#momentum = (-1, 0, 0)

	# Old code:
	'''
	if x_position is not None:
		position = (x_position * m, 0., 0.)
	else:
		position = (0 * m, 0., 0.)
	g4 = G4Generator()
	out_ph1 = g4.generate('e-', position, momentum, scintillator, gen, energy=2.)		# NOTE the energy here.  for testing !!!!!
	g4 = None
	'''

	e_x_distances = np.linspace(6.97, 7.04, 29)    # (6.97, 7.04, 29)		# (6.99, 7.001, 12)
	gamma_x_distances = np.linspace(5.5, 7.20, 35)    # (5.5, 7.20, 35)					# (5.8, 7.05, 26)

	particles = ['e-','gamma'] # ['gamma','e-']

	counts = {}
	scint_counts = {}
	cherenkov_counts = {}
	for particle in particles:
		counts[particle] = {}
		scint_counts[particle] = {}
		cherenkov_counts[particle] = {}

	print("Starting photon generation")

	run_count = 10
	for counter in xrange(run_count):
		for particle in particles:
			x_distances = e_x_distances if particle == "e-" else gamma_x_distances
			for x in x_distances:
				if counter == 0:
					counts[particle][x] = []
					scint_counts[particle][x] = []
					cherenkov_counts[particle][x] = []
				position = (x * m, 0., 0.)
				# gen = g4gen.G4Generator(scint)
				g4 = G4Generator()		# Should not be necessary
				output = g4.generate(particle, position, momentum, scintillator, gen)
				g4 = None
				counts[particle][x].append(len(output.pos))

				# Count subtypes
				subtype_bins = np.bincount(output.process_subtypes)
				# Magic numbers.  For subtype definitions, see:
				#    http://geant4.web.cern.ch/geant4/collaboration/working_groups/electromagnetic/
				scint_count = 0
				if (len(subtype_bins)) >= 22:
					scint_count = subtype_bins[22]
				scint_counts[particle][x].append(scint_count)

				cherenkov_count = 0
				if (len(subtype_bins)) >= 21:
					cherenkov_count = subtype_bins[21]
				cherenkov_counts[particle][x].append(cherenkov_count)
				if counts[particle][x][-1] != scint_counts[particle][x][-1] + cherenkov_counts[particle][x][-1]:
					print("===>>> Uh oh: counts don't add up: ", particle, x, counts[particle][x], scint_counts[particle][x], cherenkov_counts[particle][x]);

	'''
	Initial manual binning:
	        process_counters = {}
        for ptype in process_types:
            if ptype in process_counters:
                process_counters[ptype] += 1
            else:
                process_counters[ptype] = 1

        process_subtype_counters = {}
        for ptype in process_subtypes:
            if ptype in process_subtype_counters:
                process_subtype_counters[ptype] += 1
            else:
                process_subtype_counters[ptype] = 1
	'''

	e_avgs, e_yerr = compute_stats(counts['e-'], e_x_distances)
	gamma_avgs, gamma_yerr = compute_stats(counts['gamma'], gamma_x_distances)
	e_scint_avgs, _ = compute_stats(scint_counts['e-'], e_x_distances)
	gamma_scint_avgs, _ = compute_stats(scint_counts['gamma'], gamma_x_distances)
	e_cheren_avgs, _ = compute_stats(cherenkov_counts['e-'], e_x_distances)
	gamma_cheren_avgs, _ = compute_stats(cherenkov_counts['gamma'], gamma_x_distances)

	# See: https://matplotlib.org/1.2.1/examples/pylab_examples/errorbar_demo.html
	fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
	fig.suptitle("Photon Counts | 2 MeV particles fired in +X direction | " + str(run_count) + " particles per position", fontsize=14)
	plt.title("Photon counts")
	plt.xlabel('X Distance')
	plt.ylabel('Photon Count')

	# See https://stackoverflow.com/questions/12957582/matplotlib-plot-yerr-xerr-as-shaded-region-rather-than-error-bars
	# for error shading
	plt1 = axs[0]
	plt1.set_title('e -')
	plt1.grid(True)		# See: https://matplotlib.org/examples/pylab_examples/axes_props.html
	plt1.stackplot(e_x_distances, e_scint_avgs, e_cheren_avgs)
	plt1.legend(["Scintillation","Cherenkov"])
	'''
	plt1.errorbar(e_x_distances, e_avgs, yerr=e_yerr, color='#CC4F1B', capsize=3)
	#plt1.plot(e_x_distances, e_y, color='#CC4F1B')
	plt1.fill_between(e_x_distances, e_avgs - e_yerr, e_avgs + e_yerr, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
	'''

	plt2 = axs[1]
	plt2.set_title('gamma')
	plt2.grid(True)

	plt2.stackplot(gamma_x_distances, gamma_scint_avgs, gamma_cheren_avgs)
	plt2.legend(["Scintillation","Cherenkov"])

	'''
	#plt2.xlabel('X Distance')
	#plt2.ylabel('Photon Count')
	plt2.errorbar(gamma_x_distances, gamma_avgs, yerr=gamma_yerr, capsize=3)
	plt2.fill_between(gamma_x_distances, gamma_avgs - gamma_yerr, gamma_avgs + gamma_yerr, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF')
	'''
	plt.show()


	# This is how to reset the seed between runs
	#Geant4.HepRandom.setTheSeed(9876)

	print
	print("G4 state: ", Geant4.gStateManager.GetCurrentState())

	# I think we need to reset in between - one seems to affect the other???  And if reverse the order...
	# But cant just regenerate the detector: Geant4 kernel is not PreInit state : Method ignored.
	#	Then seg fault

	'''
	plt.hist(out_ph1.wavelengths,bins=400,histtype='step',linewidth=2,label='e$^-$')
	plt.hist(out_ph2.wavelengths,bins=400,histtype='step',linewidth=2,label='$\gamma$')
	plt.xlabel('scintillation photons wavelength [nm]')
	plt.legend()
	plt.show()
	plt.close()
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(out_ph1.pos[:,0],out_ph1.pos[:,1],out_ph1.pos[:,2],'.',label='e$^-$')
	ax.plot(out_ph2.pos[:,0],out_ph2.pos[:,1],out_ph2.pos[:,2],'.',label='$\gamma$')
	plt.legend()
	plt.show()
	'''

	# Trying to dump Geant4 state to start over - really dump Geant4 entirely and reload
	print("g4gen ref count: ", sys.getrefcount(g4gen))
	# gen = None
	'''
	print("g4gen ref count: ", sys.getrefcount(g4gen))
	del sys.modules["chroma.generator.g4gen"]		# This doesn't actually show in teh list - so why does this work???
	del sys.modules["chroma.generator.Geant4"]
	del sys.modules["chroma.generator.g4py"]
	del sys.modules["chroma.generator"]
	del g4gen
	'''

	#Geant4.gApplyUICommand("/run/abort")
	#Geant4.gApplyUICommand("/run/geometryModified")
	'''
	g4 = G4Generator()
	out_ph1 = g4.generate('e-', (0. * m, 0., 0.), momentum, scintillator, gen, energy=2.)		# NOTE the energy here.  for testing !!!!!
	gen.reset_g4()
	gen = None
	gen2 = g4gen.G4Generator(scintillator)
	out_ph1 = g4.generate('e-', (0. * m, 0., 0.), momentum, scintillator, gen2, energy=2.)		# NOTE the energy here.  for testing !!!!!
	'''
	final = 1
