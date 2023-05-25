import pickle
import pandas as pd

cyclo = True
gdb = False

if cyclo:
    cyclo_df = pd.read_csv("data/cyclo/full_dataset.csv", index_col=0)

    with open("data/maps_cyclo.pkl", "rb") as f:
        maps_cyclo = pickle.load(f)

    keep_cyclo_energies = []
    keep_cyclo_smiles = []
    indices = []
    # list of dictionaries
    for i, entry in enumerate(maps_cyclo):
        map = entry['mapped_rxn']
        conf = entry['confidence']
        label = cyclo_df.iloc[i]['G_act']
        if conf > 0.7:
            keep_cyclo_smiles.append(map)
            keep_cyclo_energies.append(label)
            indices.append(i)

    keep_cyclo_df = pd.DataFrame({"original idx":indices, "rxn_smiles":keep_cyclo_smiles, "barrier":keep_cyclo_energies})
    keep_cyclo_df.to_csv("data/cyclo/auto_mapped_rxns.csv")
    job_df = keep_cyclo_df[['rxn_smiles', 'barrier']]
    job_df.to_csv("data/cyclo/submit_rxns.csv", index=False)

if gdb:
    gdb_df = pd.read_csv('data/gdb7-22-ts/ccsdtf12_dz.csv', index_col=0)

    with open("data/maps_gdb.pkl", "rb") as f:
        maps_gdb = pickle.load(f)

    keep_gdb_energies = []
    keep_gdb_smiles = []
    indices = []
    for i, entry in enumerate(maps_gdb):
        map = entry['mapped_rxn']
        conf = entry['confidence']
        label = gdb_df.iloc[i]['dE0']
        if conf > 0.7:
            keep_gdb_smiles.append(map)
            keep_gdb_energies.append(label)
            indices.append(i)
    keep_gdb_df = pd.DataFrame({"original idx":indices, "rxn_smiles":keep_gdb_smiles, "barrier":keep_gdb_energies})
    keep_gdb_df.to_csv("data/gdb7-22-ts/auto_mapped_rxns.csv")

    job_df = keep_gdb_df[['rxn_smiles', 'barrier']]
    job_df.to_csv("data/gdb7-22-ts/submit_rxns.csv", index=False)
  #  print(keep_gdb_df)