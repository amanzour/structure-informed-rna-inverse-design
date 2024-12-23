
# Downloading and Preparing Data
Download a pre-processed [`processed.pt`](https://people.sunypoly.edu/~manzoua/data/processed.pt) file and [`processed_df.csv`](https://people.sunypoly.edu/~manzoua/data/processed_df.csv) metadata, and place them into the `data/` directory. Also download different generated splits [`seqid_split.pt`](https://people.sunypoly.edu/~manzoua/data/seqid_split.pt) and [`structsim_v2_split.pt`](https://people.sunypoly.edu/~manzoua/data/structsim_v2_split.pt) into `data/`. Or you can just copy all content of https://people.sunypoly.edu/~manzoua/data/ into `data/`.

**Note**
The pre-processed files refer to all RNA structures from the PDB at ≤4A resolution (~15K pdb files) downloaded via [RNASolo](https://rnasolo.cs.put.poznan.pl) on 1 August 2024. The text file `2024-08-01-pdbs.txt` contains the names of corresponding pdb files that were processed, here.
Processed pdb files have similar features as [gRNAde](https://github.com/chaitjo/geometric-rna-design) with minor additions. Secondary structure (both canonical and non-canonical) base pairs were augmented.(see below tagged as `augmented`). The software: [`X3DNA`](https://x3dna.org/) was used to generate base pairs. This software requires obtaining a license. An alternative is the freely available software  [`fr3d`](https://www.bgsu.edu/research/rna/software/fr3d.html). For the interested reader, here is a command for obtaining base pairs:
`python fr3d/classifiers/NA_pairwise_interactions.py --output . -c basepair,basepair_detail,coplanar,stacking,backbone input.pdb`. One would have to make alterations to the source code to read the corresponding format. We actually tried fr3d as well and it led to somewhat similar performance.



Each RNA will be processed into the following format (most of the metadata is optional for simply using gRNAde):
```
{
    'sequence'                   # RNA sequence as a string
    'id_list'                    # list of PDB IDs
    'coords_list'                # list of structures, i.e. 3D coordinates of shape ``(length, 27, 3)``
    'sec_struct_list'            # list of secondary structure strings in dotbracket notation
    'sec_bp_list'                # list of lists. Each list consist of base-pair tuples using x3dna (augmented)
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
The precise procedures for creating the splits (``seqid_split.pt` and `structsim_v2_split.pt`) can be found in the `notebooks/` directory and are identical to those for [gRNAde](https://github.com/chaitjo/geometric-rna-design).
