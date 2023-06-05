import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

# read results
cyclo_true_df = pd.read_csv('results/cyclo_true/test_scores.csv')
cyclo_rxnmapper_df = pd.read_csv('results/cyclo_rxnmapper/test_scores.csv')
cyclo_random_df = pd.read_csv('results/cyclo_random/test_scores.csv')
cyclo_maes = np.load('data/cyclo/slatm_10fold.npy')

gdb_true_df = pd.read_csv('results/gdb_true/test_scores.csv')
gdb_rxnmapper_df = pd.read_csv('results/gdb_rxnmapper/test_scores.csv')
gdb_random_df = pd.read_csv('results/gdb_random/test_scores.csv')
gdb_maes = np.load('data/gdb7-22-ts/slatm_10fold.npy')

proparg_random_df = pd.read_csv('results/proparg/test_scores.csv')
proparg_maes = np.load('data/proparg/slatm_10fold.npy')

#matplotlib.rcParams["figure.figsize"] = (10, 4.4)
matplotlib.rcParams.update({"font.size":14})

fig, axes = plt.subplots(nrows=1, ncols=3)

ax = axes[1]
ax.set_title('Cyclo-23-TS')
labels = ['CGR True', 'CGR RXNMapper', 'CGR Random', 'SLATM+KRR']
for i, df in enumerate([cyclo_true_df, cyclo_rxnmapper_df, cyclo_random_df]):
    ax.bar(i, df['Mean mae'], yerr=df['Standard deviation mae'])
ax.bar(3, np.mean(cyclo_maes), yerr=np.std(cyclo_maes))
ax.set_xticks(list(range(len(labels))))
ax.set_xticklabels(labels, rotation=90)

ax = axes[0]
ax.set_ylabel("MAE $\Delta G^\ddag$ [kcal/mol]")
ax.set_title('GDB7-22-TS')
for i, df in enumerate([gdb_true_df, gdb_rxnmapper_df, gdb_random_df]):
    ax.bar(i, df['Mean mae'], yerr=df['Standard deviation mae'])
ax.bar(3, np.mean(gdb_maes), yerr=np.std(gdb_maes))
ax.set_xticks(list(range(len(labels))))
ax.set_xticklabels(labels, rotation=90)

ax = axes[2]
ax.set_title("Proparg-21-TS")
ax.bar(0, proparg_random_df['Mean mae'], yerr=df['Standard deviation mae'], color='C2')
ax.bar(1, np.mean(proparg_maes), yerr=np.std(proparg_maes), color='C3')
ax.set_xticks([0,1])
ax.set_xticklabels(['CGR Random', 'SLATM+KRR'], rotation=90)

figname = 'figures/atom_mapping_quality.pdf'

plt.tight_layout()
plt.savefig(figname)