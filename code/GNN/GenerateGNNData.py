from Bio.PDB import PDBParser
import io
import torch
import numpy as np
from torch_geometric.data import Data

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
aa2onehot = {aa: np.eye(len(AMINO_ACIDS))[i] for i, aa in enumerate(AMINO_ACIDS)}

def pdb_to_graph(pdb_file, ptm_sites=None, dist_thresh=8.0):
    parser = PDBParser(QUIET=True)

    # --- Fix: open file with correct encoding
    with open(pdb_file, "r", encoding="utf-8", errors="ignore") as f:
        structure = parser.get_structure("prot", io.StringIO(f.read()))

    residues, coords, seq = [], [], []
    for res in structure.get_residues():
        hetflag, resseq, icode = res.id
        if hetflag != " " or "CA" not in res:  # skip non-standard residues
            continue
        residues.append(res)
        seq.append(res.get_resname())
        coords.append(res["CA"].coord)

    coords = np.array(coords)
    n = len(coords)
    if n == 0:
        raise ValueError("No residues with CA atoms found in file.")

    # --- Convert 3-letter to one-hot
    mapping = {
        "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
        "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
        "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
        "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"
    }
    aa1 = [mapping.get(a.upper(), "X") for a in seq]
    x = torch.tensor([aa2onehot.get(a, np.zeros(len(AMINO_ACIDS))) for a in aa1],
                     dtype=torch.float)

    # --- Edges: residues within dist_thresh Å
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    edges = np.argwhere((dist < dist_thresh) & (dist > 0))
    edge_index = torch.tensor(edges.T, dtype=torch.long)

    # --- PTM labels
    y = torch.zeros(n, dtype=torch.long)
    if ptm_sites:
        pdb_nums = [res.id[1] for res in residues]
        for site in ptm_sites:
            if site in pdb_nums:
                y[pdb_nums.index(site)] = 1

    data = Data(x=x, edge_index=edge_index, y=y)
    data.coords = torch.tensor(coords, dtype=torch.float)
    return data








pdb_path = "Downloads/1QK1.pdb"  # AKT1 AlphaFold model
ptm_sites = [308, 473]  # phosphorylation sites
graph = pdb_to_graph(pdb_path, ptm_sites, dist_thresh=8.0)

print(graph)
