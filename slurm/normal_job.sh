#!/bin/sh
#parse pubchem molecules and construct 3d from SMILES.
#SBATCH -J pubchem
#SBATCH -o pubchem.out
#SBATCH -p cpuQ --qos=normal
#SBATCH -N 1 -n 48

python mp.py
