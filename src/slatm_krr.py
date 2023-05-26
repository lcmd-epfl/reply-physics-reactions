import numpy as np
import src import reaction_reps
import src import learning
import os
import argparse as ap

def argparse():
    parser = ap.ArgumentParser()
    parser.add_argument('-c', '--cyclo', action='store_true')
    parser.add_argument('-g', '--gdb', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argparse()
    cyclo = args.cyclo
    gdb = args.gdb

    qml_obj = reaction_reps.QML()
    if cyclo:
        qml_obj.get_cyclo_data()

        if not os.path.exists('data/cyclo/slatm.npy'):
            slatm = qml_obj.get_SLATM()
            np.save('data/cyclo/slatm.npy', slatm)
        else:
            slatm = np.load('data/cyclo/slatm.npy')

        y = qml_obj.barriers
        folds = 10
        fname = 'data/cyclo/slatm_' + str(folds) +'fold.npy'
        if not os.path.exists(fname):
            maes = learning.KFold_MAE(slatm, y, folds=folds)
            np.save(fname, maes)
        else:
            maes = np.load(fname)
        mean_mae = np.mean(maes)
        std_mae = np.std(maes)
        print(f"Cyclo MAE={mean_mae}+-{std_mae}")

    if gdb:
        qml_obj.get_GDB7_ccsd_data()

        if not os.path.exists('data/gdb7-22-ts/slatm.npy'):
            slatm = qml_obj.get_SLATM()
            np.save('data/gdb7-22-ts/slatm.npy', slatm)
        else:
            slatm = np.load('data/gdb7-22-ts/slatm.npy')

        y = qml_obj.barriers
        folds = 10
        fname = 'data/gdb7-22-ts/slatm_' + str(folds) + 'fold.npy'
        if not os.path.exists(fname):
            maes = learning.KFold_MAE(slatm, y, folds=folds)
            np.save(fname, maes)
        else:
            maes = np.load(fname)
        mean_mae = np.mean(maes)
        std_mae = np.std(maes)
        print(f"GDB MAE={mean_mae}+-{std_mae}")