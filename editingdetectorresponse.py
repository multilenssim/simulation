from chroma.transform import make_rotation_matrix, normalize
from kabamland import find_inscribed_radius, find_focal_length, return_values
from ShortIO.root_short import PDFRootWriter, PDFRootReader, ShortRootReader
import detectorconfig
import numpy as np
import lensmaterials as lm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DetectorResponse(object):
    def __init__(self, configname):
        config = detectorconfig.configdict[configname]
        self.edge_length, self.facecoords, self.direction, self.axis, self.angle, self.spin_angle = return_values(config.edge_length, config.base)
        self.pmtxbins = config.pmtxbins
        self.pmtybins = config.pmtybins
        self.npmt_bins = 20*self.pmtxbins*self.pmtybins
        self.diameter_ratio = config.diameter_ratio
        self.thickness_ratio = config.thickness_ratio
        self.focal_length = find_focal_length(config.edge_length, config.base, config.diameter_ratio, config.thickness_ratio)
        self.pmt_side_length = np.sqrt(3)*(3-np.sqrt(5))*self.focal_length + self.edge_length
        self.inscribed_radius = find_inscribed_radius(self.edge_length)
        self.inverse_rotation_matrices = self.build_inverse_rotation_matrices()
        self.displacement_matrix = self.build_displacement_matrix()
        self.rotated_displacement = self.build_rotated_displacement_matrix()
        self.pdfs = [[] for i in range(self.npmt_bins)]
        self.calc1 = self.pmtxbins/self.pmt_side_length
        self.calc2 = self.pmtxbins/2.0
        self.calc3 = 2*self.pmtybins/(np.sqrt(3)*self.pmt_side_length)
        self.calc4 = self.pmtybins/3.0
        self.calc5 = self.pmtxbins*self.pmtybins
        self.totalxbins = np.empty(0)
        self.totalybins = np.empty(0)
        self.totalzbins = np.empty(0)
        self.totalpmtbins = np.empty(0)
        
    def build_inverse_rotation_matrices(self):
        rotation_matrices = np.empty((20, 3, 3))
        for k in range(20):
            rotation_matrices[k] = np.dot(make_rotation_matrix(self.spin_angle[k], self.direction[k]), make_rotation_matrix(self.angle[k], self.axis[k]))
        return np.linalg.inv(rotation_matrices)

    def build_displacement_matrix(self):
        displacement_matrix = np.empty((20, 3))
        for k in range(20):
            displacement_matrix[k] = self.facecoords[k] + self.focal_length*normalize(self.facecoords[k])
        return displacement_matrix

    def build_rotated_displacement_matrix(self):
        rotated_displacement = np.empty((20, 3))
        for k in range(20):
            rotated_displacement[k] = np.dot(self.inverse_rotation_matrices[k], self.displacement_matrix[k])
        return rotated_displacement

    def find_pmt_bin_array(self, pos_array):
        # returns an array of pmt bins corresponding to an array of end-positions
        length = np.shape(pos_array)[0]
        facebin_array = np.empty(length)
        pmt_positions = np.empty((length, 3))
        bin_array = np.empty(length)
        transpose_pos_array = np.transpose(pos_array)
        
        # finding which pmt face each position belongs to by seeing which inverse rotation and displacement brings the position closest to z = 0
        for k in range(20):
            initial_position_array = np.transpose(np.dot(self.inverse_rotation_matrices[k], transpose_pos_array)) - self.rotated_displacement[k]
            for i in range(length):
                initial_zcoord = initial_position_array[i,2]

                if (initial_zcoord < 1e-5) and (initial_zcoord > -1e-5):
                    facebin_array[i] = k
                    pmt_positions[i] = initial_position_array[i]
                    break
        xbin_array = np.floor(self.calc1*pmt_positions[:,0] + self.calc2)
        ybin_array = np.floor(self.calc3*pmt_positions[:,1] + self.calc4)
        
        #making a single bin index from the x, y and facebins.
        bin_array = facebin_array*self.calc5 + ybin_array*self.pmtxbins + xbin_array
        #returning -1 for any impossible bins
        for i in range(length):
            if (xbin_array[i] >= self.pmtxbins) or (xbin_array[i] < 0) or (ybin_array[i] >= self.pmtybins) or (ybin_array[i] < 0):
                bin_array[i] = -1
                print "bin " + str(i) + " is a culprit"
        return bin_array.astype(int)

    def build_pdfs(self, detectorxbins, detectorybins, detectorzbins, filename):
        # creates a list of a pdf for each pmt describing the likelihood of photons from particular beginning bins to record hits at that pmt
        reader = ShortRootReader(filename)
        #detectorhits = np.zeros((detectorxbins, detectorybins, detectorzbins))
        calc6x = detectorxbins/(2*self.inscribed_radius)
        calc6y = detectorybins/(2*self.inscribed_radius)
        calc6z = detectorzbins/(2*self.inscribed_radius)
        calc7x = detectorxbins/2.0
        calc7y = detectorybins/2.0
        calc7z = detectorzbins/2.0
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
                
            if np.size(self.totalpmtbins) == 0:
                self.totalxbins = xinitbin_array
                self.totalybins = yinitbin_array
                self.totalzbins = zinitbin_array
                self.totalpmtbins = pmt_bin_array
            else:
                self.totalxbins = np.concatenate((self.totalxbins, xinitbin_array))
                self.totalybins = np.concatenate((self.totalybins, yinitbin_array))
                self.totalzbins = np.concatenate((self.totalzbins, zinitbin_array))
                self.totalpmtbins = np.concatenate((self.totalpmtbins, pmt_bin_array))
                  
            #detectorbin = (zinitbin*detectorxbins*detectorybins + yinitbin*detectorxbins + xinitbin).astype(int)
            # adding counts of the photons to their respective positions in the pdf. The pdfs are normalized afterwards
            for i in range(length):
                pmt_bin = pmt_bin_array[i]
                if pmt_bin == -1:
                    culprit_count += 1
                    continue
                elif self.pdfs[pmt_bin] == []:
                    self.pdfs[pmt_bin] = np.zeros((detectorxbins, detectorybins, detectorzbins))
                    self.pdfs[pmt_bin][xinitbin_array[i], yinitbin_array[i], zinitbin_array[i]] = 1
                    count += 1
                else:
                    self.pdfs[pmt_bin][xinitbin_array[i], yinitbin_array[i], zinitbin_array[i]] += 1
            loops += 1
            print loops
            # if loops == 50:
            #     break
        
        self.normalize_pdfs()
        print "count " + str(count)
        print "there are " + str(culprit_count) + " culprits"
        self.create_3dplot(2632, detectorxbins, detectorybins, detectorzbins)
                
    def normalize_pdfs(self):
        for i in range(self.npmt_bins):
            if self.pdfs[i] != []:
                self.pdfs[i] = np.float32(self.pdfs[i]/float(np.sum(self.pdfs[i])))
        
    def create_3dplot(self, pmt_choice, nxbins, nybins, nzbins):
         selections = np.where(self.totalpmtbins == pmt_choice)[0]
         array_length = np.size(selections)
         xbins = np.empty(array_length)
         ybins = np.empty(array_length)
         zbins = np.empty(array_length)
         sizes = np.empty(array_length)
         for i in range(array_length):
             index = selections[i]
             xbins[i] = self.totalxbins[index]
             ybins[i] = self.totalybins[index]
             zbins[i] = self.totalzbins[index]
             sizes[i] = 10000*self.pdfs[int(pmt_choice)][xbins[i], ybins[i], zbins[i]]
         xs = (2*xbins-1)*self.inscribed_radius/(2.0*nxbins)
         ys = (2*ybins-1)*self.inscribed_radius/(2.0*nybins)
         zs = (2*zbins-1)*self.inscribed_radius/(2.0*nzbins)
         
         fig = plt.figure()
         ax = fig.add_subplot(111, projection='3d')
         ax.scatter(xs, ys, zs, s=sizes)
         ax.set_xlabel('X')
         ax.set_ylabel('Y')
         ax.set_zlabel('Z')
         plt.title('Initial Position of Photons for Bin ' + str(pmt_choice))
         plt.show()

    def plot_pdf(self, pdf, plot_title):
        nonzero_entries = np.nonzero(pdf)
        xs = nonzero_entries[0]
        ys = nonzero_entries[1]
        zs = nonzero_entries[2]
        sizes = 10000*pdf[xs,ys,zs]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs, s=sizes)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('PDF (adding model) of ' + str(plot_title))
        plt.show()

    def analyze_event(self, detectorxbins, detectorybins, detectorzbins, eventfile):
        ##takes an event and constructs the pdf based upon which pmts were hit.
        reader = ShortRootReader(eventfile)
        for ev in reader:
            detected = (ev.photons_end.flags & (0x1 <<2)).astype(bool) 
            ending_photons = ev.photons_end.pos[detected]
            event_pmt_bin_array = self.find_pmt_bin_array(ending_photons)
            length = np.shape(ending_photons)[0]
            print length
            
            final_pdf = np.zeros((detectorxbins, detectorybins, detectorzbins))
            for i in range(length):
                event_pmt_bin = event_pmt_bin_array[i]
                final_pdf = np.add(final_pdf, self.pdfs[event_pmt_bin])
            final_pdf = np.float32(final_pdf/float(np.sum(final_pdf)))
            print np.sum(final_pdf)
            final_pdf2 = np.float32(final_pdf/float(length))
            print np.array_equal(final_pdf,final_pdf2)
            print final_pdf
            self.plot_pdf(final_pdf, eventfile)
        return final_pdf

    def write_to_ROOT(self, filename):
        writer = PDFRootWriter(filename)
        for i in range(self.npmt_bins):
            writer.write_event(self.pdfs[i], i)
        writer.close()

    def read_from_ROOT(self, filename):
        reader = PDFRootReader(filename)
        for bin_ind, pdf in reader:
            self.pdfs[bin_ind] = pdf
            


