from rxnmapper import RXNMapper
import pandas as pd
from rdkit import Chem
import pickle

def clear_atom_map(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol

def reset_smiles(rxn_smiles):
  #  print('rxn smiles', rxn_smiles)
    reactants, products = rxn_smiles.split('>>')
    split_reactants = reactants.split('.')
    n_reactants = len(split_reactants)
    reactant_mols = [Chem.MolFromSmiles(x) for x in split_reactants]
    reactant_mols = [clear_atom_map(x) for x in reactant_mols]
    reactant_smiles = [Chem.MolToSmiles(x) for x in reactant_mols]
    assert len(reactant_smiles) == n_reactants, 'missing reactants'
    product_mol = Chem.MolFromSmiles(products)
    product_mol = clear_atom_map(product_mol)
    product_smiles = Chem.MolToSmiles(product_mol)
    mod_rxn_smiles = '.'.join(reactant_smiles) + '>>' + product_smiles
  #  print('mod', mod_rxn_smiles)
    return mod_rxn_smiles

def get_maps_and_confidence(list_rxn_smiles):
    """
    :param list_rxn_smiles:
    :return: list of dictionary of mapped_rxn and confidence
    """
    mapper = RXNMapper()
    results = mapper.get_attention_guided_atom_maps(list_rxn_smiles)
    return results

cyclo_df = pd.read_csv('data/cyclo/full_dataset.csv', index_col=0)
rxn_smiles = cyclo_df['rxn_smiles']
mod_rxn_smiles = rxn_smiles.apply(reset_smiles).to_list()
maps = get_maps_and_confidence(mod_rxn_smiles)

with open('data/maps_cyclo.pkl', 'wb') as f:
    pickle.dump(maps, f)

print("File saved")