## GNN-RT
- The transfer learning of [GNN-RT](https://github.com/Qiong-Yang/GNN-RT) effectively solves the problem of different retention times for the same compound in different chromatographic systems, and we suggest using GNN-RT for retention time prediction of candidate molecules.
- We use the retention times of known molecules that are in the same chromatographic system as the unknown molecule for transfer learning. This results in a model applicable to the retention time prediction of the chromatographic system of the unknown molecule. The model is used to predict the retention time of the molecule for the candidate molecules obtained after MZ as well as CCS screening.
- The code for all the above steps can be found on github GNN-RT repository.
