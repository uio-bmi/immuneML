# tmscoring
Python implementation of the [TMscore][2] program to compare structures of the same protein.

## Usage:
We provide three classes, `TMscoring`, `Sscoring`, and `RMSDscoring`, that only differ in their default
optimisation score.

They are initialised with the file paths to two PDB files:

```
alignment = tmscoring.TMscoring('structure1.pdb', 'structure2.pdb')

# Find the optimal alignment
alignment.optimise()

# Get the TM score:
alignment.tmscore(**alignment.get_current_values())

# Get the TM local scores:
alignment.tmscore_samples(**alignment.get_current_values())

# RMSD of the protein aligned according to TM score
alignment.rmsd(**alignment.get_current_values())

# Returns the transformation matrix between both structures:
alignment.get_matrix(**alignment.get_current_values())

# Save the aligned files:
alignment.write(outputfile='aligned.pdb', appended=True)
```

The structures can be matched by index (default), or performing a global sequence alignment with Smith-Waterman
using a match score of 2, mismatch of -1, a gap penalty of -0.5 for opening and -0.1 for extending.



### Utility functions:

`get_tm(path_to_pdb1, path_to_pdb2)` and `get_rmsd(pdb1, pdb2)` are simple wrappers that compute TM score or RMSD.


## What is different?
tmscoring is a Python library that conveniently exposes all the necessary variables.
This removes the necessity to parse files.

Also, the minimisation engine is [MINUIT's Migrad][1], a powerful and robust derivative-free minimisation algorithm,
heavily tested by particle physicists for decades.
In our testing, `tmscoring` yields the same or slightly better scores than upstream `TMscore`.


[1]: https://root.cern.ch/root/html534/TMinuit.html
[2]: https://zhanglab.ccmb.med.umich.edu/TM-score/
