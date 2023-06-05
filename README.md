# Exploring atom mapping quality for predicting reaction barriers
This repo provides the necessary code and data to reproduce the results in the reply to comment on "Physics-based representations for machine learning properties of chemical reactions".
This paper as well as the associated code explore <b>the dependency of reaction barrier prediction to the atom-mapping quality</b>.

## Data 
The datasets focussed on here are the Cyclo-23-TS, GDB7-22-TS and Proparg-21-TS. For each dataset, the following information is available 
- The xyz structures of reactants and products, and computed barriers 
- Atom maps for the original-mapped reactions (where available), RXNMapper reactions, and randomly mapped reactions.
- Corresponding submission csv files for chemprop
- The results of a 10-fold CV run of physics-based model SLATM+KRR

## Codes 
The codes in `src/` provide:
- Means to atom map the reactions using RXNMapper, using the files `mapper.py` and `process_maps.py`
- Random mappings are generated using `random_mapper.py`
- The files `learning.py` and `reaction_reps.py` are support files for `slatm_krr.py` which performs the 10-fold CV prediction with 80/10/10 splits of the SLATM+KRR model

## CGR results
Results of the CGR runs are saved to `results/`.