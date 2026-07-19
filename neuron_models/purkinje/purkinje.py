import os
import sys
import json
import pickle
import numpy as np
import neuron
from neuron import h
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm, TwoSlopeNorm
import seaborn as sns
import dbbs_models
from dbbs_models import build_purkinje_cell
pc=build_purkinje_cell()
rec={'time':h.vector(),'soma(0.5)':h.vector()}
rec['time'].record(h._ref_t)
rec['soma(0.5)'.record(pc.soma[0](0.5)._ref_v)
