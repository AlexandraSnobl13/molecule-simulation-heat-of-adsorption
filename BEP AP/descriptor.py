import torch
import torch.nn as nn
import numpy as np
import rdkit

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
from rdkit.Chem import rdFreeSASA


def get_descriptors(mol : Chem.Mol, return_dict = False):
    
    # Basic Descriptors
    mol = Chem.AddHs(mol)  # Add hydrogen atoms

    # Generate a 3D conformer
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    mol_wt = Descriptors.MolWt(mol)
    num_atoms = Descriptors.HeavyAtomCount(mol)
    
    # Surface Area
    radii = rdFreeSASA.classifyAtoms(mol)
    sasa = rdFreeSASA.CalcSASA(mol, radii)

    # LogP and Polar Surface Area
    logP = Crippen.MolLogP(mol)
    psa = Descriptors.TPSA(mol)

    # Hydrogen Bond Donors and Acceptors
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)

    # Partial Charges
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    partial_charges = np.std([float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()])


    # Collecting descriptors
    descriptor_dict = {
        'molecular_weight': mol_wt,
        'num_atoms': num_atoms,
        'surface_area': sasa,
        'logP': logP,
        'polar_surface_area': psa,
        'hydrogen_bond_donors': hbd,
        'hydrogen_bond_acceptors': hba,
        'partial_charges': partial_charges,
    }

    if return_dict:
        return descriptor_dict
    
    decriptors = torch.tensor([mol_wt, num_atoms, sasa, logP, psa, hbd, hba, partial_charges], dtype=torch.float32)

    return decriptors
 
class DescriptorModel(nn.Module):
 
    def __init__(self):
 
        super(DescriptorModel, self).__init__()
 
        smiles_strings = {
        'co2': 'O=C=O',
        'ch4': 'C',
        'c2h6': 'CC',
        'c3h8': 'CCC',
        'h2': '[H][H]'
        }
 
        molecules = {name: Chem.MolFromSmiles(smiles) for name, smiles in smiles_strings.items()}
        self.map = nn.ParameterDict({name: nn.Parameter(get_descriptors(mol), requires_grad=False)
                                     for name, mol in molecules.items()})
 
    def forward(self, batch, label = None):
       
        emb = torch.stack([self.map[l] for l in label])
        return emb
    
    def forward2(self, x, edge_index, edge_attr,
                batch, label = None):
       
        return self.forward(batch, label)