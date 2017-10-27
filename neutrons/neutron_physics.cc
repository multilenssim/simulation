#include "neutron_physics.hh"

// Probably don't need all of these
#include <G4OpticalPhysics.hh>
#include <G4EmPenelopePhysics.hh>
#include <G4Event.hh>

#include <G4HadronElasticProcess.hh>
#include <G4NeutronHPElasticData.hh>
#include <G4NeutronHPElastic.hh>
#include <G4NeutronHPBuilder.hh>
#include <G4ProcessManager.hh>

NeutronPhysicsList::NeutronPhysicsList():  G4VModularPhysicsList()
{
    // default cut value  (1.0mm)
    defaultCutValue = 1.0*CLHEP::mm;

    // General Physics
    //RegisterPhysics( new G4EmPenelopePhysics(0) );
    // Optical Physics
    //G4OpticalPhysics* opticalPhysics = new G4OpticalPhysics();
    //RegisterPhysics( opticalPhysics );

    /* Turn on neutron physics */
    //For example  elastic scattering below 20 MeV
    G4HadronElasticProcess* theNeutronElasticProcess = new G4HadronElasticProcess();
    // Cross Section Data set
    G4NeutronHPElasticData* theHPElasticData = new G4NeutronHPElasticData();
    theNeutronElasticProcess->AddDataSet( theHPElasticData );
    // Model
    G4NeutronHPElastic* theNeutronElasticModel = new G4NeutronHPElastic();
    theNeutronElasticProcess->RegisterMe(theNeutronElasticModel);
    std::cout << "Process created, registering it" << std::endl;

    G4ProcessManager* pmanager = G4Neutron::Neutron()->GetProcessManager();
    if (pmanager == NULL) {
        std::cout << "Neutron process manager is null" << std::endl;
         pmanager = new G4ProcessManager(G4Neutron::Neutron());
         G4Neutron::Neutron()->SetProcessManager(pmanager);  // XX This may be rediundant
    }
    std::cout << "Think we have neutrons: " << pmanager << std::endl;
    //pmanager->AddDiscreteProcess( theNeutronElasticProcess );

    pmanager = G4Neutron::Neutron()->GetProcessManager();
    std::cout << "== Neutron process manager ==" << std::endl;
    pmanager->DumpInfo();
    std::cout << "=============================" << std::endl;

    auto /*G4PTblDicIterator*/ *theParticleIterator = GetParticleIterator();
    theParticleIterator->reset();
    while( (*theParticleIterator)() ){
        G4ParticleDefinition* particle = theParticleIterator->value();
        G4ProcessManager* pmanager = particle->GetProcessManager();
        std::cout << "We have a pmanager: " << pmanager << std::endl;

        G4String particleName = particle->GetParticleName();
        std::cout << "Particle name: " << particleName << std::endl;
        if (theNeutronElasticProcess->IsApplicable(*particle)) {
            std::cout << "Elastic process is applicable to: " << particleName << std::endl;
            if (particleName == "neutron") {
                theNeutronElasticProcess->DumpPhysicsTable(*particle);
                pmanager->AddProcess(theNeutronElasticProcess);
                //pmanager->SetProcessOrderingToLast(theNeutronElasticProcess, idxAtRest);
                //pmanager->SetProcessOrderingToLast(theNeutronElasticProcess, idxPostStep);
                pmanager->DumpInfo();
                break;
            }
        }
    }
    pmanager = G4Neutron::Neutron()->GetProcessManager();
    std::cout << "== Neutron process manager ==" << std::endl;
    pmanager->DumpInfo();
    std::cout << "=============================" << std::endl;
}

NeutronPhysicsList::~NeutronPhysicsList()
{
}

#include <boost/python.hpp>
#include <pyublas/numpy.hpp>

using namespace boost::python;

void NeutronPhysicsList::SetCuts(){
  //  " G4VUserPhysicsList::SetCutsWithDefault" method sets
  //   the default cut value for all particle types
  SetCutsWithDefault();
}


void export_NeutronPhysics()
{
  class_<NeutronPhysicsList, NeutronPhysicsList*, bases<G4VModularPhysicsList>, boost::noncopyable > ("NeutronPhysicsList", "Neutron physics list")
    .def(init<>());

}

BOOST_PYTHON_MODULE(neutron_physics)
{
  export_NeutronPhysics();
}
