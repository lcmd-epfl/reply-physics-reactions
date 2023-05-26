import numpy as np
import pandas as pd
import qml
from glob import glob
from periodictable import elements
import os 

pt = {}
for el in elements:
    pt[el.symbol] = el.number

def convert_symbol_to_ncharge(symbol):
    return pt[symbol]

def pad_indices(idx):
    idx = str(idx)
    if len(idx) < 6: 
        pad_len = 6 - len(idx)
        pad = '0'*pad_len
        idx = pad + idx
    return idx

def check_alt_files(list_files):
    files = []
    if len(list_files) < 3:
        return list_files
    for file in list_files:
        if "_alt" in file:
            dup_file_label = file.split("_alt.xyz")[0]
    for file in list_files:
        if dup_file_label in file:
            if "_alt" in file:
                files.append(file)
        else:
            files.append(file)
    return files

def create_mol_obj(atomtypes, ncharges, coords):
    if len(atomtypes) == 0:
        return None
    mol = qml.Compound()
    mol.atomtypes = atomtypes
    mol.nuclear_charges = ncharges
    mol.coordinates = coords
    return mol

def reader(xyz):
    if not os.path.exists(xyz):
        return [], [], []
    with open(xyz, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    try:
        nat = int(lines[0])
    except:
        print('file', xyz, 'is empty')
        return [], [], [] 
    start_idx = 2
    end_idx = start_idx + nat

    atomtypes = []
    coords = []

    for line_idx in range(start_idx, end_idx):
        line = lines[line_idx]
        atomtype, x, y, z = line.split()
        atomtypes.append(str(atomtype))
        coords.append([float(x), float(y), float(z)])

    ncharges = [convert_symbol_to_ncharge(x) for x in atomtypes]

    assert len(atomtypes) == nat
    assert len(coords) == nat
    assert len(ncharges) == nat
    return np.array(atomtypes), np.array(ncharges), np.array(coords)

class QML:
    def __init__(self):
        self.ncharges = []
        self.unique_ncharges = []
        self.max_natoms = 0
        self.atomtype_dict = {"H": 0, "C": 0, "N": 0, "O": 0, "S": 0, "Cl":0,
                                "F":0}
        self.mols_products = []
        self.mols_reactants = [[]]
        return

    def get_GDB7_ccsd_data(self):
        df = pd.read_csv("data/gdb7-22-ts/ccsdtf12_dz.csv")
        self.barriers = df['dE0'].to_numpy()
        indices = df['idx'].apply(pad_indices).to_list()

        r_mols = []
        p_mols = []
        for idx in indices:
            filedir = 'data/gdb7-22-ts/xyz/'+idx
            rfile = filedir + '/r' + idx + '.xyz'
            r_atomtypes, r_ncharges, r_coords = reader(rfile)
            r_coords = r_coords * 0.529177 # bohr to angstrom
            r_mol = create_mol_obj(r_atomtypes, r_ncharges, r_coords)
            r_mols.append([r_mol])

            # multiple p files
            pfiles = glob(filedir+'/p*.xyz')
            sub_pmols = []
            for pfile in pfiles:
                p_atomtypes, p_ncharges, p_coords = reader(pfile)
                p_coords = p_coords * 0.529177 
                p_mol = create_mol_obj(p_atomtypes, p_ncharges, p_coords)
                sub_pmols.append(p_mol)
            p_mols.append(sub_pmols)
        self.mols_reactants = r_mols
        self.mols_products = p_mols
        all_r_mols = np.concatenate(r_mols)
        self.ncharges = [x.nuclear_charges for x in all_r_mols]
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges))
        return


    def get_cyclo_data(self):
        df = pd.read_csv("data/cyclo/full_dataset.csv", index_col=0)
        self.barriers = df['G_act'].to_numpy()
        indices = df['rxn_id'].to_list()
        self.indices = indices
        rxns = ["data/cyclo/xyz/"+str(i) for i in indices]
        
        reactants_files = []
        products_files = []
        for rxn_dir in rxns:
            reactants = glob(rxn_dir+"/r*.xyz")
            reactants = check_alt_files(reactants)
            assert len(reactants) == 2, f"Inconsistent length of {len(reactants)}"
            reactants_files.append(reactants)
            products = glob(rxn_dir+"/p*.xyz")
            products_files.append(products)

        mols_reactants = []
        mols_products = []
        ncharges_products = []
        for i in range(len(rxns)):
            mols_r = []
            mols_p = []
            ncharges_p = []
            for reactant in reactants_files[i]:
                mol = qml.Compound(reactant)
                mols_r.append(mol)
            for product in products_files[i]:
                mol = qml.Compound(product)
                mols_p.append(mol)
                ncharges_p.append(mol.nuclear_charges)
            ncharges_p = np.concatenate(ncharges_p)
            ncharges_products.append(ncharges_p)
            mols_reactants.append(mols_r)
            mols_products.append(mols_p)
        self.ncharges = ncharges_products
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges, axis=0))
        self.mols_reactants = mols_reactants
        self.mols_products = mols_products
        return

    def get_cyclo_xtb_data(self):
        # test set at lower quality geometry
        df = pd.read_csv("data/cyclo/full_dataset.csv", index_col=0)
        barriers = df['G_act'].to_numpy()

        filedir = 'data/cyclo/xyz-xtb/'
        files = glob(filedir+"*.xyz")
        indices = np.unique([x.split("/")[-1].split("_")[1].strip('.xyz') for x in files])
        self.indices = [int(x) for x in indices]
        reactants_files = []
        products_files = []

        r_mols = []
        p_mols = []
        barriers = []
        for i, idx in enumerate(indices):
            idx = str(idx)
            # multiple r files
            if os.path.exists(filedir+"Reactant_"+idx+".xyz"):
                rfile = filedir+"Reactant_"+idx+".xyz"
                r_atomtypes, r_ncharges, r_coords = reader(rfile)
                r_mol = create_mol_obj(r_atomtypes, r_ncharges, r_coords)
                sub_rmols.append(r_mol)
            elif os.path.exists(filedir+"Reactant_"+idx+"_0.xyz"):
                rfiles = glob(filedir+'Reactant_'+idx+'_*.xyz')
                sub_rmols = []
                for rfile in rfiles:
                    r_atomtypes, r_ncharges, r_coords = reader(rfile)
                    r_mol = create_mol_obj(r_atomtypes, r_ncharges, r_coords)
                    sub_rmols.append(r_mol)
            else:
                print("cannot find rfile")
                sub_rmols.append([None])

            sub_pmols = []
            pfile = filedir+"Product_"+idx+".xyz"
            p_atomtypes, p_ncharges, p_coords = reader(pfile)
            p_mol = create_mol_obj(p_atomtypes, p_ncharges, p_coords)
            sub_pmols.append(p_mol)

            if None not in sub_pmols and None not in sub_rmols: 
                r_mols.append(sub_rmols)
                p_mols.append(sub_pmols)
                barrier = df[df['rxn_id'] == int(idx)]['G_act'].item()
                barriers.append(barrier)
            else:
                print("skipping r mols", sub_rmols, 'and p mols', sub_pmols, 'for idx', idx) 

        assert len(r_mols) == len(p_mols)
        assert len(r_mols) == len(barriers)
        self.mols_reactants = r_mols
        self.mols_products = p_mols
        self.barriers = barriers
        all_r_mols = np.concatenate(r_mols)
        self.ncharges = [x.nuclear_charges for x in all_r_mols]
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges))
        return

    def get_SLATM(self):
        mbtypes = qml.representations.get_slatm_mbtypes(self.ncharges)


        slatm_reactants = [
            np.array(
                [
                    qml.representations.generate_slatm(
                        x.coordinates, x.nuclear_charges, mbtypes, local=False
                    )
                    for x in reactants
                ]
            )
            for reactants in self.mols_reactants
        ]

        slatm_reactants_sum = np.array([sum(x) for x in slatm_reactants])
        slatm_products = [
            np.array(
                [
                    qml.representations.generate_slatm(
                        x.coordinates, x.nuclear_charges, mbtypes, local=False
                    )
                    for x in products
                ]
            )
            for products in self.mols_products
        ]
        slatm_products = np.array([sum(x) for x in slatm_products])
        slatm_diff = slatm_products - slatm_reactants_sum

        return slatm_diff

