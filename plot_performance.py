import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

stuyver = True

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
matplotlib.rcParams.update({"font.size":13})

fig, axes = plt.subplots(nrows=1, ncols=3)
labels = ['CGR True', 'CGR RXNMapper', 'CGR Random', 'SLATM+KRR']

ax = axes[0]
ax.set_ylabel("MAE $\Delta G^\ddag$ [kcal/mol]")
ax.set_title('(a) GDB7-22-TS')
for i, df in enumerate([gdb_true_df, gdb_rxnmapper_df, gdb_random_df]):
    ax.bar(i, df['Mean mae'], yerr=df['Standard deviation mae'])
ax.bar(3, np.mean(gdb_maes), yerr=np.std(gdb_maes))
ax.set_xticks([])

ax = axes[1]
ax.set_title('(b) Cyclo-23-TS')
for i, df in enumerate([cyclo_true_df, cyclo_rxnmapper_df, cyclo_random_df]):
    ax.bar(i, df['Mean mae'], yerr=df['Standard deviation mae'], label=labels[i])
ax.bar(3, np.mean(cyclo_maes), yerr=np.std(cyclo_maes), label=labels[3])

if stuyver:
    ax.bar(4, 2.96, label='QM-GNN True')
    ax.bar(5, 3.09, label='WLN-GNN True')
ax.set_xticks([])

ax = axes[2]
ax.set_title("(c) Proparg-21-TS")
for i in range(2):
    ax.bar(i, 0)
ax.bar(2, proparg_random_df['Mean mae'], yerr=df['Standard deviation mae'])
ax.bar(3, np.mean(proparg_maes), yerr=np.std(proparg_maes))
ax.set_xticks([])

handles, labels = axes[1].get_legend_handles_labels()
plt.tight_layout()

plt.legend(handles, labels, bbox_to_anchor=(1.6,-0.02), ncol=3)

if not stuyver:
    figname = 'figures/atom_mapping_quality.pdf'
else:
    figname = 'figures/atom_mapping_quality_stuyver.pdf'

plt.savefig(figname, bbox_inches='tight')
#plt.show()