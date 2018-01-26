from Geant4 import *

import lensmaterials as lm
from neutrons.neutron_test import *


if __name__=='__main__':
    gApplyUICommand("/run/verbose 2")
    gApplyUICommand("/event/verbose 2")
    gApplyUICommand("/tracking/verbose 2")

    # set geometry
    detector = neutron_test.MyDetectorConstruction()
    gRunManager.SetUserInitialization(detector)

    # set physics list
    physics_list = QGSP_BERT_HP()
    gRunManager.SetUserInitialization(physics_list)

    primary_generator_action = neutron_test.MyPrimaryGeneratorAction()
    gRunManager.SetUserAction(primary_generator_action)

    # Initialise
    gRunManager.Initialize()

    #gUImanager.ExecuteMacroFile('macros/raytrace.mac')
    gUImanager.ExecuteMacroFile('dawn.mac')

    gRunManager.BeamOn(50)
