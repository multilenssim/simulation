from Geant4 import *


class MyDetectorConstruction(G4VUserDetectorConstruction):
    "My Detector Construction"

    def __init__(self):
        G4VUserDetectorConstruction.__init__(self)

        self.solid = {}
        self.logical = {}
        self.physical = {}

        self.create_world(side=4000,
                          material="G4_AIR")

        self.create_orb(name="scintillator",
                             radius=200,
                             translation=[0, 0, 900],
                             material="G4_Galactic",
                             colour=[1., 1., 1., 0.1],
                             mother='world')
        '''
        self.create_cylinder(name="vacuum",
                             radius=200,
                             length=320,
                             translation=[0, 0, 900],
                             material="G4_Galactic",
                             colour=[1., 1., 1., 0.1],
                             mother='world')

        self.create_cylinder(name="upper_scatter",
                             radius=10,
                             length=0.01,
                             translation=[0, 0, 60],
                             material="G4_Ta",
                             colour=[1., 1., 1., 0.7],
                             mother='vacuum')

        self.create_cylinder(name="lower_scatter",
                             radius=30,
                             length=0.01,
                             translation=[0, 0, 20],
                             material="G4_Al",
                             colour=[1., 1., 1., 0.7],
                             mother='vacuum')

        self.create_applicator_aperture(name="apature_1",
                                        inner_side=142,
                                        outer_side=182,
                                        thickness=6,
                                        translation=[0, 0, 449],
                                        material="G4_Fe",
                                        colour=[1, 1, 1, 0.7],
                                        mother='world')

        self.create_applicator_aperture(name="apature_2",
                                        inner_side=130,
                                        outer_side=220,
                                        thickness=12,
                                        translation=[0, 0, 269],
                                        material="G4_Fe",
                                        colour=[1, 1, 1, 0.7],
                                        mother='world')

        self.create_applicator_aperture(name="apature_3",
                                        inner_side=110,
                                        outer_side=180,
                                        thickness=12,
                                        translation=[0, 0, 140],
                                        material="G4_Fe",
                                        colour=[1, 1, 1, 0.7],
                                        mother='world')

        self.create_applicator_aperture(name="apature_4",
                                        inner_side=100,
                                        outer_side=140,
                                        thickness=12,
                                        translation=[0, 0, 59],
                                        material="G4_Fe",
                                        colour=[1, 1, 1, 0.7],
                                        mother='world')

        self.create_applicator_aperture(name="cutout",
                                        inner_side=100,
                                        outer_side=120,
                                        thickness=6,
                                        translation=[0, 0, 50],
                                        material="G4_Fe",
                                        colour=[1, 1, 1, 0.7],
                                        mother='world')

        self.create_cube(name="phantom",
                         side=500,
                         translation=[0, 0, -250],
                         material="G4_WATER",
                         colour=[0, 0, 1, 0.4],
                         mother='world')
        '''

    def create_world(self, **kwargs):
        material = gNistManager.FindOrBuildMaterial(kwargs['material'])
        side = kwargs['side']

        self.solid['world'] = G4Box("world", side / 2., side / 2., side / 2.)

        self.logical['world'] = G4LogicalVolume(self.solid['world'],
                                                material,
                                                "world")

        self.physical['world'] = G4PVPlacement(G4Transform3D(),
                                               self.logical['world'],
                                               "world", None, False, 0)

        visual = G4VisAttributes()
        visual.SetVisibility(False)

        self.logical['world'].SetVisAttributes(visual)

    def create_orb(self, **kwargs):
        name = kwargs['name']
        radius = kwargs['radius']
        translation = G4ThreeVector(*kwargs['translation'])
        material = gNistManager.FindOrBuildMaterial(kwargs['material'])
        visual = G4VisAttributes(G4Color(*kwargs['colour']))
        mother = self.physical[kwargs['mother']]

        self.solid[name] = G4Orb(name, 0., radius)

        self.logical[name] = G4LogicalVolume(self.solid[name],
                                             material,
                                             name)

        self.physical[name] = G4PVPlacement(None, translation,
                                            name,
                                            self.logical[name],
                                            mother, False, 0)

        self.logical[name].SetVisAttributes(visual)


    def create_cylinder(self, **kwargs):
        name = kwargs['name']
        radius = kwargs['radius']
        length = kwargs['length']
        translation = G4ThreeVector(*kwargs['translation'])
        material = gNistManager.FindOrBuildMaterial(kwargs['material'])
        visual = G4VisAttributes(G4Color(*kwargs['colour']))
        mother = self.physical[kwargs['mother']]

        self.solid[name] = G4Tubs(name, 0., radius, length / 2., 0., 2 * pi)

        self.logical[name] = G4LogicalVolume(self.solid[name],
                                             material,
                                             name)

        self.physical[name] = G4PVPlacement(None, translation,
                                            name,
                                            self.logical[name],
                                            mother, False, 0)

        self.logical[name].SetVisAttributes(visual)

    def create_cube(self, **kwargs):
        name = kwargs['name']
        side = kwargs['side']
        translation = G4ThreeVector(*kwargs['translation'])
        material = gNistManager.FindOrBuildMaterial(kwargs['material'])
        visual = G4VisAttributes(G4Color(*kwargs['colour']))
        mother = self.physical[kwargs['mother']]

        self.solid[name] = G4Box(name, side / 2., side / 2., side / 2.)

        self.logical[name] = G4LogicalVolume(self.solid[name],
                                             material,
                                             name)

        self.physical[name] = G4PVPlacement(None, translation,
                                            name,
                                            self.logical[name],
                                            mother, False, 0)

        self.logical[name].SetVisAttributes(visual)

    def create_applicator_aperture(self, **kwargs):
        name = kwargs['name']
        inner_side = kwargs['inner_side']
        outer_side = kwargs['outer_side']
        thickness = kwargs['thickness']
        translation = G4ThreeVector(*kwargs['translation'])
        material = gNistManager.FindOrBuildMaterial(kwargs['material'])
        visual = G4VisAttributes(G4Color(*kwargs['colour']))
        mother = self.physical[kwargs['mother']]

        inner_box = G4Box("inner", inner_side / 2., inner_side / 2., thickness / 2. + 1)
        outer_box = G4Box("outer", outer_side / 2., outer_side / 2., thickness / 2.)

        self.solid[name] = G4SubtractionSolid(name,
                                              outer_box,
                                              inner_box)

        self.logical[name] = G4LogicalVolume(self.solid[name],
                                             material,
                                             name)

        self.physical[name] = G4PVPlacement(None,
                                            translation,
                                            name,
                                            self.logical[name],
                                            mother, False, 0)

        self.logical[name].SetVisAttributes(visual)

    # -----------------------------------------------------------------
    def Construct(self):  # return the world volume

        return self.physical['world']


class MyPrimaryGeneratorAction(G4VUserPrimaryGeneratorAction):
    "My Primary Generator Action"

    def __init__(self):
        G4VUserPrimaryGeneratorAction.__init__(self)

        particle_table = G4ParticleTable.GetParticleTable()

        electron = particle_table.FindParticle(G4String("e-"))
        positron = particle_table.FindParticle(G4String("e+"))
        gamma = particle_table.FindParticle(G4String("gamma"))
        neutron = particle_table.FindParticle(G4String("neutron"))

        beam = G4ParticleGun()
        beam.SetParticleEnergy(6 * MeV)
        beam.SetParticleMomentumDirection(G4ThreeVector(0, 0, -1))
        beam.SetParticleDefinition(neutron)
        beam.SetParticlePosition(G4ThreeVector(0, 0, 1005))

        self.particleGun = beam

    def GeneratePrimaries(self, event):
        self.particleGun.GeneratePrimaryVertex(event)

if __name__=='__main__':
    gApplyUICommand("/run/verbose 2")
    gApplyUICommand("/event/verbose 2")
    gApplyUICommand("/tracking/verbose 2")

    # set geometry
    detector = MyDetectorConstruction()
    gRunManager.SetUserInitialization(detector)

    # set physics list
    physics_list = QGSP_BERT_HP()
    gRunManager.SetUserInitialization(physics_list)

    primary_generator_action = MyPrimaryGeneratorAction()
    gRunManager.SetUserAction(primary_generator_action)

    # Initialise
    gRunManager.Initialize()

    #gUImanager.ExecuteMacroFile('macros/raytrace.mac')
    gUImanager.ExecuteMacroFile('dawn.mac')

    gRunManager.BeamOn(50)
