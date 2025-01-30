#  Secondary-structure-Informed RNA Inverse Design

Secondary-structure-informed RNA Inverse Design, or simply structure-informed-RNA-inverse-design, is a geometric deep learning pipeline for 3D RNA inverse design that also incorporates RNA secondary-structure information.
![](/tertiaryedges.png)

*Secondary*-structure-informed RNA Inverse Design, is a geometric deep learning pipeline for 3D RNA inverse design.
The original code and methodology were adopted from **gRNAde** [gRNAde: Geometric Deep Learning for 3D RNA inverse design](https://arxiv.org/abs/2305.14749), Chaitanya K. Joshi, Arian R. Jamasb, Ramon Viñas, Charles Harris, Simon Mathis, Alex Morehead, and Pietro Liò. gRNAde: Geometric Deep Learning for 3D RNA inverse design. *ICML Computational Biology Workshop, 2023.* analogous to [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) for protein design. 
 For more information on using original code for training, testing, and also RNA design, see [gRNAde GitHub page](https://github.com/chaitjo/geometric-rna-design). RNA backbones are featurized as geometric graphs and processed via a multi-state GNN encoder which is equivariant to 3D roto-translation of coordinates as well as conformer order. Model decoder and sequence design is similarly order-invariant. 

## Similarity to gRNAde
The code structure and most of the modules as well as parameters are adopted from gRNAde. Modules include, implementation of RNA features, inter-atom angle and orientation extraction, Geomtric Vector Perceptrons (GVPs), message-passing scheme, noise-adding procedures, auto-regressive decoder, RBF interpolation of distance adn many more. Similar to gRNAde, Structure-informed RNA design generates an RNA sequence conditioned on one or more 3D RNA backbone conformations.


## Difference from gRNAde

### Major
1. There are different edge-types (primary-structure, secondary-structure, and three-dimensional or spatial) in the input graph (above image). **Note** The software needs an independent RNA secondary-structure identification tool. It currently relies on either [x3dna-dssr](https://x3dna.org/), or [Fr3d](https://www.bgsu.edu/research/rna/software/fr3d.html), which determines all the canonical and non-canonical base pairs, i.e.,the secondary-structure edge types of the input graph, from the corresponding pdb file.  
2. Positional encoding of edges is eliminated. 
3. If multiple 3D backbones are provided, the GNN encoder treats them as separate inputs, only pooling updated node embeddings at the final stage in the decoder. The reason for this choice of model architecture is to increase expressiveness of RNAs with multiple stable structures, such as riboswtiches.

### Minor
1. Edge determination in the original gRNAde was according to k-means clustering of node locations. Here, there were three different edge types. The spatial edge types were derived similarly, except that the clustering algorithm used here was dbscan. The rational was that we were more interested in extracting relative positioning of nodes (nucleotides) in RNA pockets and this is more likely using dbscan than k-means clustering which is essentially a centroid-based clustering algorithm.
2. Rhofold and ribonanzanet components were eliminated. There is currently no built-in 3D structure validation of predictions in the metrics. 


## Installation

Instructions are taken from and according to [gRNAde GitHub page](https://github.com/chaitjo/geometric-rna-design). Only additioanl instruction is to install x3da-dssr.

In order to get started, set up a python environment by following the installation instructions below. 
We have tested gRNAde on Linux with Python 3.10.12 and CUDA 11.8 on NVIDIA A100 80GB GPUs and Intel XPUs, as well as on MacOS (CPU).
```sh
# Clone structure-informed-RNA-inverse-design repository
cd ~  # change this to your prefered download location
git clone https://github.com/amanzour/structure-informed-RNA-inverse-design.git
cd structure-informed-RNA-inverse-design

# Install mamba (a faster conda)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
source ~/.bashrc
# You may also use conda or virtualenv to create your environment

# Create new environment and activate it
mamba create -n rna python=3.10
mamba activate rna
```

Set up your new python environment, starting with PyTorch and PyG
```sh
# Install Pytorch on Nvidia GPUs (ensure appropriate CUDA version for your hardware)
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Pytorch Geometric (ensure matching torch + CUDA version to PyTorch)
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.1.2+cu118.html
```



Other compulsory dependencies
```sh
# Install other python libraries
mamba install jupyterlab matplotlib seaborn pandas biopython biotite -c conda-forge
pip install wandb gdown pyyaml ipdb python-dotenv tqdm cpdb-protein torchmetrics einops ml_collections mdanalysis MDAnalysisTests draw_rna

# Install X3DNA for secondary structure determination
cd ~/structure-informed-RNA-inverse-design/tools/
tar -xvzf x3dna-v2.4-linux-64bit.tar.gz
./x3dna-v2.4/bin/x3dna_setup
# Follow the instructions to test your installation

# Install EternaFold for secondary structure prediction
cd ~/structure-informed-RNA-inverse-design/tools/
git clone --depth=1 https://github.com/eternagame/EternaFold.git && cd EternaFold/src
make
```

Another dependency but only if you need to process new data. Note that you can just skip this section if you choose to use the already processed files. See **Download already processed files**.
```sh
# Install x3dna-dssr. This software is under copyright and must be purchased. See https://inventions.techventures.columbia.edu/technologies/dssr-an-integrated--CU20391 for making a request.
# Another option would have been to use the Find RNA 3D (fr3d) software ((https://www.bgsu.edu/research/rna/software/fr3d.html), which is free . It requires altering the configs/defeault.yaml file and setting base_pairing parameter to 'sec_fr3d_list' plus some minor code modifications. We had tried this software as well and results are similar (more on this in later versions).
cp dssr-basic-*-.zip in the tools folder
unzip dssr-basic-*.zip. This will create x3dna-dssr folder in tools directory

```

<summary>Tools needed for data pre-processing. Note that if you only want to design RNAs based on trained parameters, or if you choose to use the already processed structures for training and testing (see **Download already processed files**), data processing is not needed and you can skip this section.</summary>

```sh
# (Optional) Install CD-HIT for sequence identity clustering
mamba install cd-hit -c bioconda

# (Optional) Install US-align/qTMclust for structural similarity clustering
cd ~/structure-informed-RNA-inverse-design/tools/
git clone https://github.com/pylelab/USalign.git && cd USalign/ && git checkout 97325d3aad852f8a4407649f25e697bbaa17e186
g++ -static -O3 -ffast-math -lm -o USalign USalign.cpp
g++ -static -O3 -ffast-math -lm -o qTMclust qTMclust.cpp
```
<br>

Once your python environment is set up, create your `.env` file with the appropriate environment variables; see the .env.example file included in the codebase for reference. 
```sh
cd ~/structure-informed-RNA-inverse-design/
touch .env
```
In order to train your own models from scratch, you still need to download and process raw RNA structures from RNAsolo ([instructions below](#downloading-data)).
Second options is to use the already processed files:

## Download processed files
 Simply copy all content of https://people.sunypoly.edu/~manzoua/data/ into `data/`. Required files are a pre-processed [`processed.pt`](https://people.sunypoly.edu/~manzoua/data/processed.pt) file and [`processed_df.csv`](https://people.sunypoly.edu/~manzoua/data/processed_df.csv) metadata, as well as generated splits [`seqid_split.pt`](https://people.sunypoly.edu/~manzoua/data/seqid_split.pt) and [`structsim_v2_split.pt`](https://people.sunypoly.edu/~manzoua/data/structsim_v2_split.pt) into `data/`. **Note** you can always generate new data splits from the above processed file, via the jupyter notebooks in the `notebooks` folder.
The pre-processed files refer to all RNA structures from the PDB at ≤4A resolution (~15K 3D structures) downloaded via [RNASolo](https://rnasolo.cs.put.poznan.pl) on 1 August 2024. The text file `2024-08-01-pdbs.txt` contains the names of corresponding pdb files that were processed, here.


## Directory Structure and Usage

```
.
├── README.md
├── LICENSE
|
├── gRNAde.py                       # gRNAde python module and command line utility
├── main.py                         # Main script for training and evaluating models
|
├── .env.example                    # Example environment file
├── .env                            # Your environment file
|
├── checkpoints                     # Saved model checkpoints
├── configs                         # Configuration files directory
├── data                            # Dataset and data files directory
├── notebooks                       # Directory for Jupyter notebooks
├── tutorial                        # Tutorial with example usage
|
├── tools                           # Directory for external tools
|   ├── draw_rna                    # RNA secondary structure visualization
|   ├── EternaFold                  # RNA sequence to secondary structure prediction tool
|   └── x3dna-v2.4                  # RNA secondary structure determination from 3D
|   |__ x3dna-dssr                  # RNA base pairing determination from 3D     
|
└── src                             # Source code directory
    ├── constants.py                # Constant values for data, paths, etc.
    ├── evaluator.py                # Evaluation loop and metrics
    ├── layers.py                   # PyTorch modules for building Multi-state GNN models
    ├── models.py                   # Multi-state GNN models for gRNAde
    ├── trainer.py                  # Training loop
    |
    └── data                        # Data-related code
        ├── clustering_utils.py     # Methods for clustering by sequence and structural similarity
        ├── data_utils.py           # Methods for loading PDB files and handling coordinates
        ├── dataset.py              # Dataset and batch sampler class
        ├── featurizer.py           # Featurizer class
        └── sec_struct_utils.py     # Methods for secondary structure prediction and determination
```



## Downloading and Preparing Data

If you would like to train your own models from scratch, download and extract the raw `.pdb` files into the `data/raw/` directory (or another location indicated by the `DATA_PATH` environment variable in your `.env` file). To manually download the pdb files, go to RNAsolo > Download arhchive > select data type & format = `3D structure (PDB)`, molecule classification = `all molecules`, redundancy (equivalence classes) = `all class members`, resolution in angstroms = `< 4.0` > click on download archive.

Next, process the raw PDB files into our ML-ready format, which will be saved under `data/processed.pt`. 
You need to install the optional dependencies (US-align, CD-HIT) for processing.
```sh
# Process raw data into ML-ready format (this may take several hours)
cd ~/structure-informed-RNA-inverse-design/
python data/process_data.py
```

Each RNA will be processed into the following format (most of the metadata is optional for simply using gRNAde):
```
{
    'sequence'                   # RNA sequence as a string
    'id_list'                    # list of PDB IDs
    'coords_list'                # list of structures, i.e. 3D coordinates of shape ``(length, 27, 3)``
    'sec_struct_list'            # list of secondary structure strings in dotbracket notation
    'sec_bp_list'                # list of secondary structure base pairs in tuples on integers **This feature is added to the original list
    'sasa_list'                  # list of per-nucleotide SASA values
    'rfam_list'                  # list of RFAM family IDs
    'eq_class_list'              # list of non-redundant equivalence class IDs
    'type_list'                  # list of structure types (RNA-only, RNA-protein complex, etc.)
    'rmsds_list'                 # dictionary of pairwise C4' RMSD values between structures
    'cluster_seqid0.8'           # cluster ID of sequence identity clustering at 80%
    'cluster_structsim0.45'      # cluster ID of structure similarity clustering at 45%
}
```

## Splits for Benchmarking

The precise procedure for creating the splits (which can be used to modify and customise them) can be found in the `notebooks/` directory.

## RNA design using the gRNAde module
For desiging RNAs, you can use the already trained parameters. Parameters are available in the **checkpoints** directory. [The model trained using seqid split strategy](checkpoints/SIRD_ARv1_3state_seqid.h5) uses the data downloaded on August 1, 2024. There was a total of around 15 thousand structures were at resolution ≤ 4Å. For a comprehensive list of pdb files used for this training see [This list](data/2024-08-01-pdbs.txt).\
If you like to use a model that is only trained on solo RNAs (and RNA-DNA Hybrids), alter this line in the gRNAde.py file: CHECKPOINT_PATH = os.path.join(PROJECT_PATH, "checkpoints/SIRD_ARv1_3state_seqid.h5") to use SIRD_ARv1_3state_seqid_soloRNA.h5 instead. Save gRNAde.py. Then run tutorial.py.\
Use the **tutorial.py** file for designing sequences for a specific input PDB file. The program uses the modified version of the **gRNAde.py** module for sequence design.


## Citations

### Our preprint:
```
@article{Amir2024structinformed,
 doi = {10.20944/preprints202412.2156.v1},
 url = {https://doi.org/10.20944/preprints202412.2156.v1},
 year = 2024,
 month = {December},
 publisher = {Preprints},
 author = {Amirhossein Manzourolajdad and Mohammad Mohebbi},
 title = {Secondary-Structure-Informed RNA Inverse Design via Relational Graph Neural Networks},
 journal = {Preprints}
}
```

### Original Software by Joshi et al:
```
@article{joshi2024grnade,
 title={gRNAde: Geometric Deep Learning for 3D RNA inverse design},
 author={Joshi, Chaitanya K and Jamasb, Arian R and Vi{\~n}as, Ramon and Harris, Charles and Mathis, Simon V and Morehead, Alex and Anand, Rishabh and Li{\`o}, Pietro},
 journal={bioRxiv},
 year={2024},
 publisher={Cold Spring Harbor Laboratory Preprints}
}

@incollection{joshi2024grnade,
 title={gRNAde: A Geometric Deep Learning pipeline for 3D RNA inverse design},
 author={Joshi, Chaitanya K and Li{\`o}, Pietro},
 booktitle={RNA Design: Methods and Protocols},
 pages={121--135},
 year={2024},
 publisher={Springer}
}
```
<hr/>

## Details on differences between structure-informed and gRNAde modules

### configs/default.yaml
parameters raduis, primary_dist, and base_pairing added.

### data/process_data.py
feature ['sec_bp_list'] added to seq_to_data output. The feature contains lists of coordinates of secondary-structure base pairs of the RNA structure.

### notebooks/
file data_stats_edges.ipynb added.

### src/data/data_utils.py
functions pdb_to_tensor_2, get_twist, and dist_2 added.

### src/data/dataset.py
class BatchSampler(data.Sampler) altered to feed RNA files one by one to the encoder.

### src/data/featurizer.py
file heavily altered.\
class class RNAGraphFeaturizer(object) altered: \
    parameter edges_s altered from having dimensions [num_edges, num_conf, num_bb_atoms x num_rbf + num_posenc + num_bb_atoms] to dimensions [num_edges, num_conf, num_bb_atoms x num_rbf + num_bb_atoms].\
    parameter num_posenc eliminated.\
    parameteres primary_dist = 500 and base_pairing added.\
    parameter radius repurposed.\
following parameters added to data output:\
    edge_s_list = edge_s_list, # num_conf x num_edges x (num_bb_atoms x num_rbf + num_bb_atoms)\
    edge_v_list = edge_v_list, # num_conf x num_edges x num_bb_atoms x 3\
    edge_index_list = edge_index_list,    # list of 2 x num_edges\
    p_s_t_map_list = p_s_t_map_list,      # primary/secondary/spatial edge types\
    **Note** unlike gRNAde, different conformations of the RNA do not have identical number of edges. Therefore, edges are saved in lists as opposed to tensors.\
functions featurize_from_pdb_file and featurize_from_pdb_filelist altered.\
functions get_angle and offset_basepairs added.\
function unpaired_cluster_dbscan added and used instead of torch_cluster.knn_graph.\
function get_k_random_entries_and_masks_2 added and used intead of get_k_random_entries_and_masks.\

### src/data/sec_struct_utils.py
constant DSSR_PATH added.\
functions get_unpaired, pdb_to_sec_struct_bp, pdb_to_x3dna_2, x3dna_to_sec_struct_2, and fr3d_to_sec_struct added.

### src/data/constants.py
constant DSSR_PATH added.

### src/data/evaluator.py
function evaluate altered to eliminate metrics 'sc_score_ribonanzanet' and 'sc_score_rhofold'.

### src/data/layers.py
class MultiGVPConvLayer(nn.Module) and its forward function restructured to handle multiple edge types.\
class MultiGVPConv(MessagePassing) and its forward function restructured to handle new edge dimensions.\
functions _split_multi and _merge_multi restructured to handle new node and edge dimensions.

### src/data/models.py
class AutoregressiveMultiGNNv1(torch.nn.Module), its forward and sample functions all altered to accommodate for multiple iterations to feed multiple structures and to take into account different edge types in within each iteration.\
class NonAutoregressiveMultiGNNv1(torch.nn.Module), its forward and sample functions all altered similarly.\
function pool_multi_conf_2 added and used instead of pool_multi_conf.

### src/data/trainer.py
function train altered to eliminate metrics 'sc_score_ribonanzanet' and 'sc_score_rhofold'.

### tools/
directories rhofold and ribonanzanet removed.

### gRNAde.py
dictionary CHECKPOINT_PATH altered to only contain trained paramters file checkpoints/SIRD_ARv1_3state_seqid.h5.\
EDGE_IN_DIM altered from (131, 3) to (99, 3).\
class class gRNAde(object) altered to generate correct RNAGraphFeaturizer object.

### main.py
function main altered to exclude metrics 'sc_score_ribonanzanet' and 'sc_score_rhofold'.\
function get_dataset altered to generate correct RNADesignDataset object.

### tutorial.py
file repurposed for RNA inverse design.
