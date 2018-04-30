from chroma.geometry import Material
from chroma.geometry import Surface

from Geant4.hepunit import *

import numpy as np

ls_refractive_index = 1.5
lensmat_refractive_index = 2.0
lensmat_ohara_refractive_index = 1.923

# This is elemental helium - vs. G4_HE
he = Material('He')
he.set('refractive_index', 1.0)
he.set('absorption_length', 1e8)
he.set('scattering_length', 1e8)
he.density = 0.5
he.composition = { 'He' : 1.0 }

lensmat = Material('lensmat')
lensmat.set('refractive_index', lensmat_refractive_index)
lensmat.set('absorption_length', 1e8)
lensmat.set('scattering_length', 1e8)

lensmat_ohara = Material('lensmat_ohara')
lensmat_ohara.set('refractive_index', lensmat_ohara_refractive_index)
lensmat_ohara.set('absorption_length', 1e8)
lensmat_ohara.set('scattering_length', 1e8)

blackhole = Material('blackhole')
blackhole.set('refractive_index', 1.0)
blackhole.set('absorption_length', 1e-15)
blackhole.set('scattering_length', 1e8)

#creates a surface with complete detection-- used on pmt
fulldetect = Surface('fulldetect')
fulldetect.set('detect', 1.0)

#creates a surface with complete absorption-- used on volume boundary (previously flat pmt detecting surface)
fullabsorb = Surface('fullabsorb')
fullabsorb.set('absorb', 1.0)

noreflect = Surface('noreflect')
#noreflect.transmissive = 1.0
noreflect.model = 2

mirror = Surface('mirror')
mirror.set('reflect_specular', 1.0)

# myglass = Material('glass')
# myglass.set('refractive_index', 1.49)
# myglass.absorption_length = \
#     np.array([(200, 0.1e-6), (300, 0.1e-6), (330, 1000.0), (500, 2000.0), (600, 1000.0), (770, 500.0), (800, 0.1e-6)])
# myglass.set('scattering_length', 1e8)


def k_thresh(r_idx,m=0.511):
# return the Cerenkov threshold energy (kinetic) for a given particle (mass in MeV) crossing media with r_idx as refractive index
	s = r_idx*r_idx
	return m*(1/np.sqrt(1.0-1.0/s)-1.0)

_ls = None

# TODO: Many modules rely on lm.ls which will have to be changed
# "create" is not really a great name for this.  Use "get" or perhaps no prefix?
def create_scintillation_material():
	global _ls
	if _ls is None:
		# ls stands for "liquid scintillator"
		_ls = Material('liquid-scintillator')
		_ls.set('refractive_index', ls_refractive_index)
		_ls.set('absorption_length', 1e8)
		_ls.set('scattering_length', 1e8)
		_ls.density = 0.780
		_ls.composition = {'H': 0.663210, 'C': 0.336655, 'N': 1.00996e-4, 'O': 3.36655e-5}

		#_ls.set('refractive_index', np.linspace(ls_refractive_index, ls_refractive_index, 38))

		# Scintillation properties
		energy_scint = list((2 * pi * hbarc / (np.linspace(320, 300, 11).astype(float) * nanometer)))
		spect_scint = list([0.04, 0.07, 0.20, 0.49, 0.84, 1.00, 0.83, 0.55, 0.40, 0.17, 0.03])
		# See https://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch02s03.html
		# Need to validate that the types are being passed through properly.  Previously was using list(Scnt_PP.astype(float)
		_ls.set_scintillation_property('FASTCOMPONENT', energy_scint, spect_scint)

		# TODO: These keys much match the Geant4 pmaterial property names.  (Magic strings)
		_ls.set_scintillation_property('SCINTILLATIONYIELD', 8000. / MeV)  # Was 10000 originally
		_ls.set_scintillation_property('RESOLUTIONSCALE', 1.0)  # Was 1.0 originally
		_ls.set_scintillation_property('FASTTIMECONSTANT', 1. * ns)
		_ls.set_scintillation_property('YIELDRATIO', 1.0)  # Was 0.8 - I think this is all fast

		# Required for scintillation yield per particle
		# Commenting out these three lines causes a "malloc: *** error for object 0x1298c02d0: pointer being freed was not allocated" WTF?
		electron_energy_scint = list([0., 1. * MeV, 10. * MeV, 100. * MeV])
		electron_yield = list([0, 8000., 80000., 800000.])
		_ls.set_scintillation_property('ELECTRONSCINTILLATIONYIELD', electron_energy_scint, electron_yield)
		proton_energy_scint = list([0 * MeV, 0.1 * MeV, 0.5 * MeV, 1 * MeV, 1.5 * MeV, 2 * MeV, 2.5 * MeV, 3 * MeV, 3.5 * MeV, 4 * MeV, 4.4 * MeV, 5 * MeV, 6 * MeV, 7 * MeV, 8 * MeV, 9 * MeV, 10 * MeV, 12 * MeV, 14 * MeV, 16 * MeV, 18 * MeV, 19.9])
		proton_yield = list([0, 43, 581, 1637, 2988, 4562, 6316, 8219, 10249, 12389, 14171, 16946, 21809, 26921, 32240, 37736, 43383, 55059, 67150, 79575, 92277, 104558])
		_ls.set_scintillation_property('PROTONSCINTILLATIONYIELD', proton_energy_scint , proton_yield)

	return _ls
