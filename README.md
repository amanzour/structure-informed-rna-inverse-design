#  Secondary-structure-Informed RNA Inverse Design

Secondary-structure-informed RNA Inverse Design, or simply structure-informed-RNA-inverse-design, is a geometric deep learning pipeline for 3D RNA inverse design that also incorporates RNA secondary-structure information.
![](/tertiaryedges.png)

*Secondary*-structure-informed RNA Inverse Design, is a geometric deep learning pipeline for 3D RNA inverse design that also incorporates RNA secondary-structure information.
The original code and methodology were adopted from **gRNAde** [gRNAde: Geometric Deep Learning for 3D RNA inverse design](https://arxiv.org/abs/2305.14749), Chaitanya K. Joshi, Arian R. Jamasb, Ramon Viñas, Charles Harris, Simon Mathis, Alex Morehead, and Pietro Liò. gRNAde: Geometric Deep Learning for 3D RNA inverse design. *ICML Computational Biology Workshop, 2023.* analogous to [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) for protein design. 
 For more information on using original code for training, testing, and also RNA design, see [gRNAde GitHub page](https://github.com/chaitjo/geometric-rna-design).

## Similarity to gRNAde
Similar to gRNAde, Structure-informed RNA design generates an RNA sequence conditioned on one or more 3D RNA backbone conformations.
RNA backbones are featurized as geometric graphs and processed via a multi-state GNN encoder which is equivariant to 3D roto-translation of coordinates as well as conformer order. Model decoder and sequence design is similarly order-invariant. 

## Difference from gRNAde

### Major
1. There are different edge-types (primary-, secondary-, and tertiary-structure) in the input graph (above image). **Note** The software needs an independent RNA secondary-structure identification tool. It currently relies on [x3dna-dssr](https://x3dna.org/), which reads the corresponding pdb file(s) and determines all the canonical and non-canonical base pairs, i.e.,the secondary-structure edge types of the input graph. 
2. Positional encoding of edges is eliminated. 
3. If multiple 3D backbones are provided, the GNN encoder treats them as separate inputs, only pooling updated node embedings at the final stage in the decoder. The reason for this choice of model architecture is to increase expressiveness of RNAs with multiple stable structures, such as riboswtiches.

### Minor
1. Edge determination in the original `gRNAde` was according to `k-means` clustering of node locations. Here, there were three different edge types. The tertiary-structure edge types were done similar to the roginal method. However, the clustering algorithm used here was `dbscan`. The rational was that we were more interested in extracting relative positioning of nodes (nucleotides) in cavities.
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
# Another option would have been to use the Find RNA 3D (fr3d) software ((https://www.bgsu.edu/research/rna/software/fr3d.html), which is free . It requires some more modification to the source code. We had tried this software as well and results are similar (more on this in later versions).
cp dssr-basic-*-.zip in the tools folder
unzip dssr-basic-*.zip. This will create x3dna-dssr folder in tools directory

```

<summary>Tools needed for data pre-processing. Note that if you only want to design RNAs based on trained parameters, or if you choose to use the already processed structures (see **Download already processed files**), data processing is not needed and you can skip this section.</summary>

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

You're now ready to use econdary-structure-informed RNA Inverse Design. To design sequences based on a 3D structure, use the **tutorial.py** file, which uses the modified version of the **gRNAde.py** module.
In order to train your own models from scratch, you still need to download and process raw RNA structures from RNAsolo ([instructions below](#downloading-data)).
Second options is to use the already processed files:

## Download processed files
 Simply copy all content of https://people.sunypoly.edu/~manzoua/data/ into `data/`. Required files are a pre-processed [`processed.pt`](https://people.sunypoly.edu/~manzoua/data/processed.pt) file and [`processed_df.csv`](https://people.sunypoly.edu/~manzoua/data/processed_df.csv) metadata, as well as generated splits [`seqid_split.pt`](https://people.sunypoly.edu/~manzoua/data/seqid_split.pt) and [`structsim_v2_split.pt`](https://people.sunypoly.edu/~manzoua/data/structsim_v2_split.pt) into `data/`. **Note** you can always generate new data splits from the above processed file, via jupyter notebooks in the `notebooks` folder.
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
For desiging RNAs, you can use the already trained parameters. Parameters are available in the **checkpoints** directory. [The model trained using seqid split strategy](checkpoints/SIRD_ARv1_3state_seqid.h5) uses the data downloaded on August 1, 2024. There was a total of around 15 thousand structures were at resolution ≤ 4Å. For a comprehensive list of pdb files used for this training see [This list](data/2024-08-01-pdbs.txt).


## Citations

### Original Software:
```
@article{joshi2023grnade,
  title={gRNAde: Geometric Deep Learning for 3D RNA inverse design},
  author={Joshi, Chaitanya K. and Jamasb, Arian R. and Vi{\~n}as, Ramon and Harris, Charles and Mathis, Simon and Morehead, Alex and Anand, Rishabh and Li{\`o}, Pietro},
  journal={arXiv preprint},
  year={2023},
}
```
