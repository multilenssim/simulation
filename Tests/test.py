import os, os.path
import shutil
import numpy as np
import chroma.event as event
from chroma.tools import count_nonzero
from chroma.rootimport import ROOT
ROOT.gSystem.Load('libCint')

chroma_dir = os.path.expanduser('~/.chroma')
home_root_C = os.path.join(chroma_dir, 'root.C')
ROOT.gROOT.ProcessLine('.L '+home_root_C+'+')

t = ROOT.TFile("test.root")
T = t.Get("T")

nentries = T.GetEntries()
for n in range(nentries):
    T.GetEntry(n)
    ev = T.ev

    print ev

raw_input('e')
