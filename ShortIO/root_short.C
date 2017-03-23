#include <TVector3.h>
#include <vector>
#include <TTree.h>
#include <string>
// Copied from root.C; for writing shortened versions of events, with only photon positions

struct Photon_short {
  virtual ~Photon_short() { };

  TVector3 pos;
  unsigned int flag;

  ClassDef(Photon_short, 1);
};

struct Event_short {
  virtual ~Event_short() { };

  int id;
  std::vector<Photon_short> photons_beg;
  std::vector<Photon_short> photons_end;

  ClassDef(Event_short, 1);
};

struct PMT_pdf {
  virtual ~PMT_pdf() { };

  int pmt_bin_ind;
  int detector_bins_x;
  int detector_bins_y;
  int detector_bins_z;
  std::vector<float> counts;
  

  ClassDef(PMT_pdf, 1);
};

struct PMT_angles {
  virtual ~PMT_angles() { };

  int pmt_bin_ind;
  int detector_bins_x;
  int detector_bins_y;
  int detector_bins_z;
  std::vector<TVector3> angles;
  //std::vector<float> angles;

  ClassDef(PMT_angles, 1);
};

struct Gauss_angle {
  virtual ~Gauss_angle() { };

  int pmt_bin_ind;
  TVector3 mean;
  float sigma;

  ClassDef(Gauss_angle, 1);
};

//Fill photon data with zeros for the parts that are not available from short event data
void get_photons(const std::vector<Photon_short> &photons, float *pos, float *dir,
		 float *pol, float *wavelengths, float *t,
		 int *last_hit_triangles, unsigned int *flags)
{
  for (unsigned int i=0; i < photons.size(); i++) {
    const Photon_short &photon = photons[i];
    pos[3*i] = photon.pos.X();
    pos[3*i+1] = photon.pos.Y();
    pos[3*i+2] = photon.pos.Z();
	
	dir[3*i] = 0.;
    dir[3*i+1] = 0.;
    dir[3*i+2] = 0.;
    
    pol[3*i] = 0.;
    pol[3*i+1] = 0.;
    pol[3*i+2] = 0.;

    wavelengths[i] = 0.;
    t[i] = 0.;
    flags[i] = photon.flag;
    last_hit_triangles[i] = -1;
  }
}
		 
void fill_photons(std::vector<Photon_short> &photons, unsigned int nphotons, float *pos, unsigned int *flags)
{
  photons.resize(nphotons);
  
  for (unsigned int i=0; i < nphotons; i++) {
    Photon_short &photon = photons[i];
    photon.pos.SetXYZ(pos[3*i], pos[3*i + 1], pos[3*i + 2]);
    photon.flag = flags[i];

  }
}

void fill_pdf(PMT_pdf &pdf, unsigned int nbins, float *bin_counts)
{
  pdf.counts.clear();
  for (unsigned int i=0; i < nbins; i++) {
    pdf.counts.push_back(bin_counts[i]);
  }
}

void fill_angles(PMT_angles &pmt_ang, unsigned int nbins, float *bin_angles)
{
  pmt_ang.angles.clear();
  for (unsigned int i=0; i < nbins; i++) {
	TVector3 ang;
	ang.SetXYZ(bin_angles[3*i], bin_angles[3*i + 1], bin_angles[3*i + 2]);
    pmt_ang.angles.push_back(ang);
	//pmt_ang.angles.push_back(bin_angles[3*i]);
  }
}

void fill_Gauss_angle(Gauss_angle &gauss_ang, float *mean, float sigma)
{
  gauss_ang.mean.SetXYZ(mean[0],mean[1],mean[2]);
  gauss_ang.sigma = sigma;
}

void get_angles(const std::vector<TVector3> &angles, float *ang_out)
{
  for (unsigned int i=0; i < angles.size(); i++) {
    const TVector3 &ang = angles[i];
    ang_out[3*i] = ang.X();
    ang_out[3*i+1] = ang.Y();
    ang_out[3*i+2] = ang.Z();
  }
}

void get_Gauss_angle(const Gauss_angle &angle, float *mean, float *sigma)
{
    mean[0] = angle.mean.X();
    mean[1] = angle.mean.Y();
    mean[2] = angle.mean.Z();
    sigma[0] = angle.sigma;
}

#ifdef __MAKECINT__
#pragma link C++ class vector<Photon_short>;
#endif


