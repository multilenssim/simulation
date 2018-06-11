#from ShortIO.root_short import PDFRootWriter, PDFRootReader, ShortRootReader
import numpy as np

from DetectorResponse import DetectorResponse

class DetectorResponsePDF(DetectorResponse):
    '''Detector calibration information is stored in 3D PDFs for each PMT: the value
    of the PDF for PMT i at detector bin j corresponds roughly to the probability 
    that a photon hitting PMT i came from detector bin j.
    '''
    def __init__(self, configname, detectorxbins=10, detectorybins=10, detectorzbins=10, infile=None):
        # If passed infile, will automatically read in the calibrated detector PDFs (and set detector bins)
        DetectorResponse.__init__(self, configname, detectorxbins, detectorybins, detectorzbins)
        self.pdfs = [[] for i in range(self.npmt_bins)]
        if infile is not None:
            self.read_from_ROOT(infile)
        
    
    def calibrate(self, simname, nevents=-1):
        # Use with a simulation file 'simname' to calibrate the detector
        # creates a list of a pdf for each pmt describing the likelihood of photons from particular beginning bins to record hits at that pmt
        self.is_calibrated = True
        reader = ShortRootReader(simname)
        #detectorhits = np.zeros((self.detectorxbins, self.detectorybins, self.detectorzbins))
        calc6x = self.detectorxbins/(2*self.inscribed_radius)
        calc6y = self.detectorybins/(2*self.inscribed_radius)
        calc6z = self.detectorzbins/(2*self.inscribed_radius)
        calc7x = self.detectorxbins/2.0
        calc7y = self.detectorybins/2.0
        calc7z = self.detectorzbins/2.0
        count = 0
        culprit_count = 0
        loops = 0
        
        for ev in reader:
            detected = (ev.photons_end.flags & (0x1 <<2)).astype(bool)
            beginning_photons = ev.photons_beg.pos[detected]
            ending_photons = ev.photons_end.pos[detected]
            length = np.shape(ending_photons)[0]
            
            #creating an array of pmt bins for the ending_photons
            pmt_bin_array = self.find_pmt_bin_array(ending_photons)

            # creating arrays for the detector bins of each photon.
            xinitbin_array = np.floor(calc6x*beginning_photons[:,0]+calc7x).astype(int)
            yinitbin_array = np.floor(calc6y*beginning_photons[:,1]+calc7y).astype(int)
            zinitbin_array = np.floor(calc6z*beginning_photons[:,2]+calc7z).astype(int)
                  
            # adding counts of the photons to their respective positions in the pdf. The pdfs are normalized afterwards
            for i in range(length):
                pmt_bin = pmt_bin_array[i]
                if pmt_bin == -1:
                    culprit_count += 1
                    continue
                elif self.pdfs[pmt_bin] == []:
                    self.pdfs[pmt_bin] = np.zeros((self.detectorxbins, self.detectorybins, self.detectorzbins))
                    self.pdfs[pmt_bin][xinitbin_array[i], yinitbin_array[i], zinitbin_array[i]] = 1
                    count += 1
                else:
                    self.pdfs[pmt_bin][xinitbin_array[i], yinitbin_array[i], zinitbin_array[i]] += 1
            loops += 1
            print loops
            if loops == nevents:
                break
            # if loops == 50:
            #     break
        
        self.normalize_pdfs()
        print "count " + str(count)
        print "there are " + str(culprit_count) + " culprits"
                
    def normalize_pdfs(self):
        # Ensure PDFs are normalized to 1
        for i in range(self.npmt_bins):
            if self.pdfs[i] != []:
                self.pdfs[i] = np.float32(self.pdfs[i]/float(np.sum(self.pdfs[i])))
                
    def write_to_ROOT(self, filename):
        # Write the PDF values to a ROOT file
        writer = PDFRootWriter(filename)
        for i in range(self.npmt_bins):
            writer.write_event(self.pdfs[i], i)
        writer.close()

    def read_from_ROOT(self, filename):
        # Read the PDF values from a ROOT file; set the detector bins shape to fit the PDF
        self.is_calibrated = True
        reader = PDFRootReader(filename)
        shape = np.shape(reader.jump_to(0)[1])
        #print "Shape: " + str(shape)
        self.detectorxbins = shape[0]
        self.detectorybins = shape[1]
        self.detectorzbins = shape[2]
        for bin_ind, pdf in reader:
            self.pdfs[bin_ind] = pdf
            if bin_ind % 1000 == 0:
                print "Read in "+str(bin_ind)+" PMT PDFs."
