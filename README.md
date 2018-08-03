## Expertise modeling for OpenReview

A key part of good paper-reviewer matching is having a good model of affinity between papers and reviewers. This repository holds code and tools for generating affinity scores between papers and reviewers.

### Tips for training models on the cluster with Slurm:

Open a screen with `screen`

Open a zsh shell on a CPU node with 24 cores and 120G of memory:

`srun --pty --mem=120G --mincpus=24 --partition=cpu /bin/zsh`

Train your model (e.g. `python train-my-model.py`)

Escape the screen with ctrl + a + d
