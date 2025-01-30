######################################################################
# Geometric RNA Design, Joshi et al.
# Original repository: https://github.com/chaitjo/geometric-rna-design
######################################################################

# Import libraries and set up the environment

import sys
# sys.path.append('../')

import dotenv
dotenv.load_dotenv(".env")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.constants import PROJECT_PATH

# Import the gRNAde module
from gRNAde import gRNAde

# Create an instance of gRNAde
gRNAde_module = gRNAde(split='das', max_num_conformers=3, gpu_id=0)

# Single-state design example usage
sequences, samples, perplexity, recovery_sample, sc_score = gRNAde_module.design_from_pdb_file(
    pdb_filepath = os.path.join(PROJECT_PATH, "[path to file]/2CKY_1_A.pdb"),
    output_filepath = os.path.join(PROJECT_PATH, "[path to file]/2CKY_1_A_test.fasta"),
    n_samples = 16,
    temperature = 1.0,
    seed = 0
)
for seq in sequences:
    print(seq.format('fasta'))
