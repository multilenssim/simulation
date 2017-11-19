#ifndef __neutron_physics_hh__
#define __neutron_physics_hh__

#include <G4VModularPhysicsList.hh>

class NeutronPhysicsList: public G4VModularPhysicsList
{
public:
  NeutronPhysicsList();
  virtual ~NeutronPhysicsList();
  virtual void AddHadronElasticProcess();
  virtual void SetCuts();
};


#endif
