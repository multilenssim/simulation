import numpy as np
import pprint
import inspect
from itertools import izip, count

from Geant4 import *
from Geant4.hepunit import *

import g4py.ezgeom
import g4py.NISTmaterials
import g4py.ParticleGun
import g4py.EMSTDpl


import neutron_physics

# all material/surface properties are interpolated at these
# wavelengths when they are sent to the gpu
standard_wavelengths = np.arange(60, 810, 20).astype(np.float32)

class Material(object):
    """Material optical properties."""
    def __init__(self, name='none'):
        self.name = name

        self.refractive_index = None
        self.absorption_length = None
        self.scattering_length = None
        self.set('reemission_prob', 0)
        self.set('reemission_cdf', 0)
        self.density = 0.0 # g/cm^3
        self.composition = {} # by mass
        self.scintillation_properties = {}

    def set(self, name, value, wavelengths=standard_wavelengths):
        if np.iterable(value):
            if len(value) != len(wavelengths):
                raise ValueError('shape mismatch')
        else:
            value = np.tile(value, len(wavelengths))

        self.__dict__[name] = np.array(zip(wavelengths, value), dtype=np.float32)

    def set_scintillation_property(self, name, value, energies=None):
        if np.iterable(value):
            if len(value) != len(energies):
                raise ValueError('shape mismatch')
            self.scintillation_properties[name] = np.array(zip(value, energies), dtype=np.float32)
        else:
            self.scintillation_properties[name] = value

# Copied form chroma
class G4Generator(object):

    def __init__(self, material, seed=None, g4_detector_parameters=None):
        self.track_tree = None
        if seed is not None:
            HepRandom.setTheSeed(seed)

        # From g4py examples/exucation/lesson1
        # ------------------------------------------------------------------
        # setup for materials
        # ------------------------------------------------------------------
        # simple materials for Qgeom
        g4py.NISTmaterials.Construct()

        # ------------------------------------------------------------------
        # setup for geometry
        # ------------------------------------------------------------------
        # g4py.Qgeom.Construct()
        g4py.ezgeom.Construct()  # initialize

        # ------------------------------------------------------------------
        # setup for physics list
        # ------------------------------------------------------------------
        g4py.EMSTDpl.Construct()

        # ------------------------------------------------------------------
        # setup for primary generator action
        # ------------------------------------------------------------------
        self.particle_gun = g4py.ParticleGun.Construct()

        #self.physics_list = _g4chroma.ChromaPhysicsList()
        #gRunManager.SetUserInitialization(self.physics_list)

        if g4_detector_parameters is None:
            world_material = self.find_material(material)
            g4py.ezgeom.SetWorldMaterial(world_material)
            g4py.ezgeom.ResizeWorld(100 * m, 100 * m, 100 * m)
            self.world = g4py.ezgeom.G4EzVolume('world')
        else:
            if g4_detector_parameters.world_material is not None:
                self.world_material = self.find_material(g4_detector_parameters.world_material)
            else:  # If no parameters at all - we should go to default behavior
                self.world_material = G4Material.GetMaterial("G4_Galactic")  # Use find?
                                #  G4 does not allow pure vacuum.  Galactic is near vacuum.

            g4py.ezgeom.SetWorldMaterial(self.world_material)
            g4py.ezgeom.ResizeWorld(100 * m, 100 * m, 100 * m)

            self.world = g4py.ezgeom.G4EzVolume('world')

            self.scintillaton_material = self.find_material(material)
            if g4_detector_parameters.orb_radius is not None:
                self.world.CreateOrbVolume(self.scintillaton_material, g4_detector_parameters.orb_radius * m)
            else:  # Don't assume that we have box_size
                self.world.CreateBoxVolume(self.scintillaton_material, g4_detector_parameters.box_size * m,
                                           g4_detector_parameters.box_size * m, g4_detector_parameters.box_size * m)

        self.world.PlaceIt(G4ThreeVector(0, 0, 0))

        # gRunManager.SetUserInitialization(self.world)   # ????

        '''
        self.event_action = g4_user_actions.ChromaEventAction()
        gRunManager.SetUserAction(self.event_action)
        # self.event_action2 = _g4chroma.EventAction()       # Can only have one
        # gRunManager.SetUserAction(self.event_action2)

        self.tracking_action = g4_user_actions.ChromaTrackingAction()
        # self.tracking_action = _g4chroma.PhotonTrackingAction()
        gRunManager.SetUserAction(self.tracking_action)
        '''
        # gRunManager.Initialize()

    def find_material(self, material):     # This could be a class method - don't need self
        g4_material = None
        if type(material) == str:
            g4_material = gNistManager.FindOrBuildMaterial(material)
        elif type(material) == Material:
            g4_material = self.create_g4material(material)
        #else:
            # Check it for this type: G4Material.GetMaterial("G4_Galactic") - i.e. allow a third option?
        if g4_material is None:
            raise StandardError("Material could not be found: " + str(material))  # Looks like nothing catches this??
        return g4_material

    def create_g4material(self, material):      # This could be a class method - don't need self
        g4material = G4Material('detector_material', material.density * g / cm3,
                                len(material.composition))

        # Add elements
        for element_name, element_frac_by_weight in material.composition.items():
            element = G4Element.GetElement(element_name, True)
            if element == None:
                print("Building element: ", element_name)
                element = gNistManager.FindOrBuildElement(element_name)    # Isotopes parameter?
            if element == None:
                print("Element not found: ", element_name)
                # Fail the method?
            else:
                g4material.AddElement(element, element_frac_by_weight)

        # Set index of refraction
        prop_table = G4MaterialPropertiesTable()
        # Reverse entries so they are in ascending energy order rather
        # than wavelength
        energy = list((2*pi*hbarc / (material.refractive_index[::-1,0] * nanometer)).astype(float))
        values = list(material.refractive_index[::-1, 1].astype(float))
        prop_table.AddProperty('RINDEX', energy, values)

        # Add scintillation properties
        # These may be single values, or a set of values for a range of energies
        # (see geometry.Material)
        try:
            for key, value in material.scintillation_properties.iteritems():
                if np.iterable(value):
                    spectrum = value[:,0].tolist()
                    energies = value[:,1].tolist()
                    prop_table.AddProperty(key, spectrum, energies)
                else:
                    prop_table.AddConstProperty(key, value)
        except AttributeError:
            pass

        # Load properties
        g4material.SetMaterialPropertiesTable(prop_table)
        return g4material

    '''
    ChromaPhysicsList::ChromaPhysicsList():  G4VModularPhysicsList()
    {
      // default cut value  (1.0mm)
      defaultCutValue = 1.0*CLHEP::mm;
    
      // General Physics
      RegisterPhysics( new G4EmPenelopePhysics(0) );
      // Optical Physics
      G4OpticalPhysics* opticalPhysics = new G4OpticalPhysics();
      RegisterPhysics( opticalPhysics );
    
      /* Turn on neutron physics */
      /*
      //For example  elastic scattering below 20 MeV
      G4HadronElasticProcess* theNeutronElasticProcess = new G4HadronElasticProcess();
      // Cross Section Data set
      G4NeutronHPElasticData* theHPElasticData = new G4NeutronHPElasticData();
      theNeutronElasticProcess->AddDataSet( theHPElasticData );
      // Model
      G4NeutronHPElastic* theNeutronElasticModel = new G4NeutronHPElastic();
      theNeutronElasticProcess->RegisterMe(theNeutronElasticModel);
      std::cout << "Process created, registering it" << std::endl;
    
      G4ProcessManager* pmanager = G4Neutron::Neutron()-> GetProcessManager();
      std::cout << "Think we have neutrons" << std::endl;
      //pmanager->AddDiscreteProcess( theNeutronElasticProcess );
    */
    /*
      G4PTblDicIterator *theParticleIterator = GetParticleIterator();
      theParticleIterator->reset();
      while( (*theParticleIterator)() ){
        G4ParticleDefinition* particle = theParticleIterator->value();
        G4ProcessManager* pmanager = particle->GetProcessManager();
        G4String particleName = particle->GetParticleName();
        if (theNeutronElasticProcess->IsApplicable(*particle)) {
          if (particleName == "neutron") {
            theNeutronElasticProcess->DumpPhysicsTable(*particle);
            pmanager->AddProcess(theNeutronElasticProcess);
            pmanager->SetProcessOrderingToLast(theNeutronElasticProcess, idxAtRest);
            pmanager->SetProcessOrderingToLast(theNeutronElasticProcess, idxPostStep);
            break;
          }
        }
      }
      */
      //theNeutronElasticProcess->DumpPhysicsTable();
    
    }
    '''



    def generate_photons(self, vertices):
        photons = None
        try:
            print("Vertex count: " + str(len(vertices)))
            for vertex in vertices:
                self.particle_gun.SetParticleByName(vertex.particle_name)
                # mass = G4ParticleTable.GetParticleTable().FindParticle(vertex.particle_name).GetPDGMass()
                total_energy = vertex.ke * MeV  # + mass
                self.particle_gun.SetParticleEnergy(total_energy)

                # Must be float type to call GEANT4 code
                pos = np.asarray(vertex.pos, dtype=np.float64)
                dir = np.asarray(vertex.dir, dtype=np.float64)

                self.particle_gun.SetParticlePosition(G4ThreeVector(*pos) * mm)
                self.particle_gun.SetParticleMomentumDirection(G4ThreeVector(*dir).unit())
                self.particle_gun.SetParticleTime(vertex.t0 * ns)

                if vertex.pol is not None:
                    self.particle_gun.SetParticlePolarization(G4ThreeVector(*vertex.pol).unit())

                #self.tracking_action.Clear()
                print('===== Turning beam on =====')
                gRunManager.BeamOn(1)
                '''
                if photons is None:
                    photons = self._extract_photons_from_tracking_action()
                else:
                    photons += self._extract_photons_from_tracking_action()
                '''
        finally:
            pass # Was used to mute / unmute....

        return photons

########## All pulled from Chroma ###########
class Vertex(object):
    def __init__(self, particle_name, pos, dir, ke, t0=0.0, pol=None):
        '''Create a particle vertex.

           particle_name: string
               Name of particle, following the GEANT4 convention.
               Examples: e-, e+, gamma, mu-, mu+, pi0

           pos: array-like object, length 3
               Position of particle vertex (mm)

           dir: array-like object, length 3
               Normalized direction vector

           ke: float
               Kinetic energy (MeV)

           t0: float
               Initial time of particle (ns)

           pol: array-like object, length 3
               Normalized polarization vector.  By default, set to None,
               and the particle is treated as having a random polarization.
        '''
        self.particle_name = particle_name
        self.pos = pos
        self.dir = dir
        self.pol = pol
        self.ke = ke
        self.t0 = t0

def norm(x):
    "Returns the norm of the vector `x`."
    return np.sqrt((x*x).sum(-1))


class Event(object):
    def __init__(self, id=0, primary_vertex=None, vertices=None, photons_beg=None, photons_end=None, channels=None):
        '''Create an event.

            id: int
              ID number of this event

            primary_vertex: chroma.event.Vertex
              Vertex information for primary generating particle.

            vertices: list of chroma.event.Vertex objects
              Starting vertices to propagate in this event.  By default
              this is the primary vertex, but complex interactions
              can be representing by putting vertices for the
              outgoing products in this list.

            photons_beg: chroma.event.Photons
              Set of initial photon vertices in this event

            photons_end: chroma.event.Photons
              Set of final photon vertices in this event

            channels: chroma.event.Channels
              Electronics channel readout information.  Every channel
              should be included, with hit or not hit status indicated
              by the channels.hit flags.
        '''
        self.id = id

        self.nphotons = None

        self.primary_vertex = primary_vertex

        if vertices is not None:
            if np.iterable(vertices):
                self.vertices = vertices
            else:
                self.vertices = [vertices]
        else:
            self.vertices = []

        self.photons_beg = photons_beg
        self.photons_end = photons_end
        self.channels = channels


def constant(obj):
    while True:
        yield obj


def isotropic():
    while True:
        yield uniform_sphere()

def flat(e_lo, e_hi):
    while True:
        yield np.random.uniform(e_lo, e_hi)

def uniform_sphere(size=None, dtype=np.double):
    """
    Generate random points isotropically distributed across the unit sphere.

    Args:
        - size: int, *optional*
            Number of points to generate. If no size is specified, a single
            point is returned.

    Source: Weisstein, Eric W. "Sphere Point Picking." Mathworld.
    """

    theta, u = np.random.uniform(0.0, 2*np.pi, size), \
        np.random.uniform(-1.0, 1.0, size)

    c = np.sqrt(1-u**2)

    if size is None:
        return np.array([c*np.cos(theta), c*np.sin(theta), u])

    points = np.empty((size, 3), dtype)

    points[:,0] = c*np.cos(theta)
    points[:,1] = c*np.sin(theta)
    points[:,2] = u

    return points


def particle_gun(particle_name_iter, pos_iter, dir_iter, ke_iter,
                 t0_iter=constant(0.0), start_id=0):
    for i, particle_name, pos, dir, ke, t0 in izip(count(start_id), particle_name_iter, pos_iter, dir_iter, ke_iter, t0_iter):
        dir = dir/norm(dir)
        vertex = Vertex(particle_name, pos, dir, ke, t0=t0)
        ev_vertex = Event(i, vertex, [vertex])
        yield ev_vertex

# Set RI to 1.0 to turn off Cherenkov light
ls_refractive_index = 1.5

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

		_ls.set('refractive_index', np.linspace(ls_refractive_index, ls_refractive_index, 38))

		''' Test code
		r_idx = 1.3
		scint = Material('scint_mat')
		scint.set('refractive_index',np.linspace(1.5,1.33,38))
		#scint.set('refractive_index',np.linspace(1.,1.,38))		# Testing inedx 1.0  = air
		scint.density = 1
		scint.composition = { 'H': 1.0/9.0, 'O': 8.0/9.0}
		'''
		#energy_ceren = list((2 * pi * hbarc / (_ls.refractive_index[::-1, 0].astype(float) * nanometer)))
		#values = list(_ls.refractive_index[::-1, 1].astype(float))
		# prop_table.AddProperty('RINDEX', energy_ceren, values)

		# Scintillation properties
		energy_scint = list((2 * pi * hbarc / (np.linspace(320, 300, 11).astype(float) * nanometer)))
		#XX energy_scint = 1.
		spect_scint = list([0.04, 0.07, 0.20, 0.49, 0.84, 1.00, 0.83, 0.55, 0.40, 0.17, 0.03])
		# See https://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch02s03.html
		# Need to validate that the types are being passed through properly.  Previously was using list(Scnt_PP.astype(float)
		_ls.set_scintillation_property('FASTCOMPONENT', energy_scint, spect_scint)
		# scint.set_scintillation_property('SLOWCOMPONENT', Scnt_PP, Scnt_SLOW);

		# This causes different effects from using the separate FAST and SLOW components below
		#  From KamLAND photocathode paper   # Per Scott
		# kabamland.detector_material.set_scintillation_property('SCINTILLATION', [float(2*pi*hbarc / (360. * nanometer))], [float(1.0)])

		# TODO: These keys much match the Geant4 pmaterial property names.  (Magic strings)

		_ls.set_scintillation_property('SCINTILLATIONYIELD', 8000. / MeV)  # Was 10000 originally
		_ls.set_scintillation_property('RESOLUTIONSCALE', 1.0)  # Was 1.0 originally
		_ls.set_scintillation_property('FASTTIMECONSTANT', 1. * ns)
		_ls.set_scintillation_property('SLOWTIMECONSTANT', 10. * ns)
		_ls.set_scintillation_property('YIELDRATIO', 1.0)  # Was 0.8 - I think this is all fast
	return _ls

if __name__=='__main__':
    gApplyUICommand("/run/verbose 2")
    gApplyUICommand("/event/verbose 2")
    gApplyUICommand("/tracking/verbose 2")

    # set geometry
    #detector = MyDetectorConstruction()
    #gRunManager.SetUserInitialization(detector)

    g4gen = G4Generator(create_scintillation_material())
    # gRunManager.SetUserInitialization(g4gen.world)  # Don't work with ezgeom??  Maybe don't need it

    # set physics list
    physics_list = QGSP_BERT_HP()
    gRunManager.SetUserInitialization(physics_list)

    # AddPhysicsList("OpticalPhysics")

    # physics_list = G4EmPenelopePhysics()
    # gRunManager.SetUserInitialization(physics_list)

    #primary_generator_action = MyPrimaryGeneratorAction()
    #gRunManager.SetUserAction(primary_generator_action)

    # Initialise
    gRunManager.Initialize()
    print('===== Geant4 initialized =====')

    ListPhysicsList()

    #gUImanager.ExecuteMacroFile('macros/raytrace.mac')
    print('===== Running macro file =====')
    #gUImanager.ExecuteMacroFile('dawn.mac')

    momentum = (1, 0, 0)
    position = (0, 0, 0)
    amount = 10
    energy = 2*MeV

    print('===== Creating Vertex =====')

    # See kabamland2
    vertex = Vertex('neutron', position, momentum, energy)
    print('===== Generating events =====')
    output = g4gen.generate_photons([vertex])

    '''
    track_tree = g4gen.track_tree
    print('===== g4gen:')
    if track_tree is not None:
        pprint.pprint(track_tree)
    if hasattr(output, "__dict__"):
        pprint.pprint(output.__dict__)
    print('===== Photon count: ' + str(len(output.dir)) + " =====")
    '''
    #gRunManager.BeamOn(50)



'''
if __name__=='__main__':
    Geant4.gApplyUICommand("/run/verbose 2")
    Geant4.gApplyUICommand("/event/verbose 2")
    Geant4.gApplyUICommand("/tracking/verbose 2")

    # set physics list
    physics_list = neutron_physics.NeutronPhysicsList()
    physics_list.AddHadronElasticProcess()
    #physics_list = QGSP_BERT_HP()  # This is from EXO200
    gRunManager.SetUserInitialization(physics_list)

    g4gen = G4Generator("G4_Pb") # Note - this initializes Geant4 (as a sort of ugly side effect)

    Geant4.gApplyUICommand("/PhysicsList/RegisterPhysics G4EmStandardPhysics")

    print('|||||||||||||||||| Find neutron in particle list ||||||||||||||||||')


    boo = gParticleTable
    #boo.DumpTable()
    last_particle = None
    print('--------')

    for particle in gParticleTable.GetParticleList():
        #particle.DumpTable()    # This duplicates the DumpTable() above
        last_particle = particle
        if particle.GetParticleName() == 'neutron':
            particle.DumpTable()
            pm = particle.GetProcessManager()
            print('===>>> Neutron process manager <<<===')
            pm.DumpInfo()
    print('|||||||||||||||||| Dump process table ||||||||||||||||||')

    goo = gProcessTable
    for i in inspect.getmembers(goo):
        print(i)

        ''
        # Ignores anything starting with underscore
        # (that is, private and protected attributes)
        if not i[0].startswith('_'):
            # Ignores methods
            if not inspect.ismethod(i[1]):
                print(i)
        ''
    print('Process table length: ' + str(goo.Length))
    ''
    list = goo.GetNameList()
    for process_name in list:
        print(process_name)
    ''
    processes = goo.FindProcesses()
    for process in processes:
        process.DumpInfo()
    print('||||||||||||||||||||||||||||||||||||')

    momentum = (1, 0, 0)
    position = (0, 0, 0)
    amount = 10
    energy = 2*MeV

    # See kabamland2
    vertex = Vertex('neutron', position, momentum, energy)
    output = g4gen.generate_photons([vertex])

    # gun = particle_gun(['neutron'] * amount, constant(position), isotropic(), flat(float(energy) * 0.99, float(energy) * 1.01))
    # output = g4gen.generate_photons(gun)
    #for ev in sim.simulate(gun, keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100):
'''