### Tips for training models on the cluster with Slurm:

Open a screen with `screen`

Open a zsh shell on a CPU node with 24 cores and 120G of memory (N,.=:

`srun --pty --mem=120G --mincpus=24 --partition=cpu /bin/zsh`

Train your model (e.g. `python train-my-model.py`)

Escape the screen with ctrl + a + d
