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

    def get_proparg_data(self):
        data = pd.read_csv("data/proparg/data.csv", index_col=0)
        reactants_files = [
            "data/proparg/data_react_xyz/"
            + data.mol.values[i]
            + data.enan.values[i]
            + ".xyz"
            for i in range(len(data))
        ]
        products_files = [
            "data/proparg/data_prod_xyz/"
            + data.mol.values[i]
            + data.enan.values[i]
            + ".xyz"
            for i in range(len(data))
        ]
        all_mols = [qml.Compound(x) for x in reactants_files + products_files]
        self.barriers = data.Eafw.to_numpy()
        self.ncharges = [mol.nuclear_charges for mol in all_mols]
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges))

        self.mols_reactants = [[qml.Compound(x)] for x in reactants_files]
        self.mols_products = [[qml.Compound(x)] for x in products_files]

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

