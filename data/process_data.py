######################################################################
# Geometric RNA Design, Joshi et al.
# Original repository: https://github.com/chaitjo/geometric-rna-design
######################################################################

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import dotenv
dotenv.load_dotenv(".env")

import os
import argparse
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import biotite
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from MDAnalysis.analysis.align import rotation_matrix
from MDAnalysis.analysis.rms import rmsd as get_rmsd

from src.data.data_utils import pdb_to_tensor, pdb_to_tensor_2, get_c4p_coords
from src.data.clustering_utils import cluster_sequence_identity, cluster_structure_similarity

import warnings
warnings.filterwarnings("ignore", category=biotite.structure.error.IncompleteStructureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


DATA_PATH = os.environ.get("DATA_PATH")


keep_insertions = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', dest='project_name', default='gRNAde-process', type=str)
    parser.add_argument('--entity', dest='entity', default='amanzour', type=str)
    parser.add_argument('--expt_name', dest='expt_name', default='process_data', type=str)
    parser.add_argument('--tags', nargs='+', dest='tags', default=[])
    parser.add_argument('--no_wandb', action="store_true")
    args, unknown = parser.parse_known_args()

    
    # Initialise wandb
    args.no_wandb = True
    if args.no_wandb:
        wandb.init(project=args.project_name, entity=args.entity, name=args.expt_name, mode='disabled')
    else:
        wandb.init(
            project=args.project_name, 
            entity=args.entity,
            name=args.expt_name, 
            tags=args.tags,
            mode='online'
        )

    print("\nLoading non-redundant equivalence class table")
    eq_class_table = pd.read_csv(os.path.join(DATA_PATH, "nrlist_3.306_4.0A.csv"), names=["eq_class", "representative", "members"], dtype=str)
    eq_class_table.eq_class = eq_class_table.eq_class.apply(lambda x: x.split("_")[2].split(".")[0])

    id_to_eq_class = {}
    eq_class_to_ids = {}
    for i, row in tqdm(eq_class_table.iterrows(), total=len(eq_class_table)):
        ids_in_class = []
        for member in row["members"].split(","):
            _member = member.replace("|", "_")
            _chains = _member.split("+")
            if len(_chains) > 1:
                _member = _chains[0]
                for chain in _chains[1:]:
                    _member += f"-{chain.split('_')[2]}"

            id_to_eq_class[_member] = row["eq_class"]
            ids_in_class.append(_member)
        
        eq_class_to_ids[row["eq_class"]] = ids_in_class

    print("\nLoading RNAsolo table")
    rnasolo_table = pd.read_csv(os.path.join(DATA_PATH, "rnasolo-main-table.csv"), dtype=str)
    rnasolo_table.eq_class = rnasolo_table.eq_class.apply(lambda x: str(x).split(".")[0])

    eq_class_to_type = {}
    for i, row in tqdm(rnasolo_table.iterrows(), total=len(rnasolo_table)):
        eq_class_to_type[row["eq_class"]] = row["molecule"]

    print("\nLoading RFAM table")
    rfam_table = pd.read_csv(os.path.join(DATA_PATH, "RFAM_families_27062023.csv"), dtype=str)

    id_to_rfam = {}
    for i, row in tqdm(rfam_table.iterrows(), total=len(rfam_table)):
        if row["pdb_id"].upper() not in id_to_rfam.keys():
            id_to_rfam[row["pdb_id"].upper()] = row["id"]

    # Initialise empty dictionaries
    id_to_seq = {}
    seq_to_data = {}
    error_ids = []

    print(f"\nProcessing raw PDB files from {DATA_PATH}")
    filenames = tqdm(os.listdir(os.path.join(DATA_PATH, "raw")))
    for filename in filenames:
        try:
            structure_id, file_ext = os.path.splitext(filename)
            filenames.set_description(structure_id)
            if file_ext != ".pdb": continue

            # if structure_id in ["357D_1_C-B-A","8G5N_1_R", "6HXX_1_Bb"]:
            #     dummy = 2
            sequence, coords, sec_struct, sasa, sec_bp = pdb_to_tensor_2(
                os.path.join(DATA_PATH, "raw", filename),
            #    os.path.join(DATA_PATH, "fr3d", structure_id + "_basepair.txt"),
                keep_insertions=keep_insertions,
                keep_pseudoknots=False
            )
            
            # basic post processing validation:
            # do not include sequences with less than 10 nucleotides,
            # which is the minimum length for sequence identity clustering
            if len(sequence) <= 10: 
                continue

            # get RFAM family
            rfam = id_to_rfam[structure_id.split("_")[0]] if \
                structure_id.split("_")[0] in id_to_rfam.keys() else "unknown"

            # get non-redundant equivalence class
            eq_class = id_to_eq_class[structure_id] if structure_id in \
                id_to_eq_class.keys() else "unknown"

            # get structure type (solo RNA, RNA-protein, RNA-DNA)
            struct_type = eq_class_to_type[eq_class] if eq_class in \
                eq_class_to_type.keys() else "unknown"

            # update dictionary    
            if sequence in seq_to_data.keys():
                # align coords of current structure to first entry
                coords_0 = seq_to_data[sequence]['coords_list'][0]
                R_hat = rotation_matrix(
                    get_c4p_coords(coords),  # mobile set
                    get_c4p_coords(coords_0) # reference set
                )[0]
                coords = coords @ R_hat.T

                # compute C4' RMSD of current structure to all other structures
                for other_id, other_coords in zip(seq_to_data[sequence]['id_list'], seq_to_data[sequence]['coords_list']):
                    seq_to_data[sequence]['rmsds_list'][(structure_id, other_id)] = get_rmsd(
                        get_c4p_coords(coords), 
                        get_c4p_coords(other_coords), 
                        superposition=True
                    )

                seq_to_data[sequence]['id_list'].append(structure_id)
                seq_to_data[sequence]['coords_list'].append(coords.float())
                seq_to_data[sequence]['sec_struct_list'].append(sec_struct)
                seq_to_data[sequence]['sec_bp_list'].append(sec_bp) #modified
                # seq_to_data[sequence]['sec_fr3d_list'].append(fr3d_bp) 
                seq_to_data[sequence]['sasa_list'].append(sasa)
                seq_to_data[sequence]['rfam_list'].append(rfam)
                seq_to_data[sequence]['eq_class_list'].append(eq_class)
                seq_to_data[sequence]['type_list'].append(struct_type)
            
            # create new entry for new sequence
            else:
                seq_to_data[sequence] = {
                    'sequence': sequence,               # sequence string
                    'id_list': [structure_id],          # list of PDB IDs
                    'coords_list': [coords.float()],    # list of 3D coordinates of shape ``(length, 27, 3)``
                    'sec_struct_list': [sec_struct],    # list of secondary structure strings
                    'sec_bp_list': [sec_bp],            # list of secondary structure base pairs
                #    'sec_fr3d_list': [fr3d_bp],
                    'sasa_list': [sasa],                # list of SASA values of shape ``(length, )``
                    'rfam_list': [rfam],                # list of RFAM family IDs
                    'eq_class_list': [eq_class],        # list of non-redundant equivalence class IDs
                    'type_list': [struct_type],         # list of structure types
                    'rmsds_list': {},                   # dictionary of pairwise C4' RMSD values between structures
                    'cluster_seqid0.8': -1,             # cluster ID of sequence identity clustering at 80%
                    'cluster_structsim0.45': -1         # cluster ID of structure similarity clustering at 45%
                }

            id_to_seq[structure_id] = sequence
        
        # catch errors and check manually later
        except Exception as e:
            print(structure_id, e)
            error_ids.append((structure_id, e))

    print(f"\nSaving (partially) processed data to {DATA_PATH}")
    torch.save(seq_to_data, os.path.join(DATA_PATH, "processed.pt"))
    
    print("\nClustering at 80% sequence similarity (CD-HIT-EST)")
    id_to_cluster_seqid = cluster_sequence_identity(
        [SeqRecord(Seq(seq), id=data["id_list"][0]) for seq, data in seq_to_data.items()],
        identity_threshold = 0.8,
    )

    unclustered_idx = max(list(id_to_cluster_seqid.values())) + 1
    for seq, data in seq_to_data.items():
        id = data["id_list"][0]
        if id in id_to_cluster_seqid.keys():
            seq_to_data[seq]['cluster_seqid0.8'] = id_to_cluster_seqid[id]
        else:
            seq_to_data[seq]['cluster_seqid0.8'] = unclustered_idx
            unclustered_idx += 1
    
    print(f"\nSaving (partially) processed data to {DATA_PATH}")
    torch.save(seq_to_data, os.path.join(DATA_PATH, "processed.pt"))

    print("\nClustering at 45% structure similarity (US-align)")
    # using first structure per sequence
    cluster_list_structsim = cluster_structure_similarity(
        [os.path.join(DATA_PATH, "raw", data["id_list"][0]+".pdb") for data in seq_to_data.values()],
        similarity_threshold = 0.45
    )

    for i, cluster in enumerate(cluster_list_structsim):
        for id in cluster:
            seq_to_data[id_to_seq[id.split(".")[0]]]['cluster_structsim0.45'] = i

    print(f"\nSaving processed data to {DATA_PATH}")
    torch.save(seq_to_data, os.path.join(DATA_PATH, "processed.pt"))

    # save to csv
    df = pd.DataFrame.from_dict(seq_to_data, orient="index", columns=["id_list", 'rfam_list', 'eq_class_list', 'type_list', 'cluster_seqid0.8', 'cluster_structsim0.45'])
    df["sequence"] = df.index
    df.reset_index(drop=True, inplace=True)
    df["length"] = df.sequence.apply(lambda x: len(x))
    df["mean_rmsd"] = df.sequence.apply(lambda x: np.mean(list(seq_to_data[x]["rmsds_list"].values()))).fillna(0.0)
    df["median_rmsd"] = df.sequence.apply(lambda x: np.median(list(seq_to_data[x]["rmsds_list"].values()))).fillna(0.0)
    df["num_structures"] = df.id_list.apply(lambda x: len(x))
    df.to_csv(os.path.join(DATA_PATH, "processed_df.csv"), index=False)

    # print IDs with errors
    if len(error_ids) > 0:
        print("\nIDs with errors (check manually):")
        for id, error in error_ids:
            print(f"{id}: {error}")
