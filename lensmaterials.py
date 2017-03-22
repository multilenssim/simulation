from chroma.geometry import Material, Solid, Surface
import numpy as np

ls_refractive_index = 1.5
lensmat_refractive_index = 2.0

#ls stands for "liquid scintillator"
ls = Material('ls')
ls.set('refractive_index', ls_refractive_index)
ls.set('absorption_length', 1e8)
ls.set('scattering_length', 1e8)
ls.density = 0.780
ls.composition = { 'H' : 0.663210, 'C' : 0.336655, 'N' : 1.00996e-4, 'O': 3.36655e-5 }

lensmat = Material('lensmat')
lensmat.set('refractive_index', lensmat_refractive_index)
lensmat.set('absorption_length', 1e8)
lensmat.set('scattering_length', 1e8)

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

