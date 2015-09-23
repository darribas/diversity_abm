# Software setup

This package and analysis can be reproduced using a conda environment with the
required dependencies installed. Dependencies are listed in `env_spec.txt`,
placed on the code folder.
Assuming you have the Anaconda Python distribution installed 
([see page](https://store.continuum.io/cshop/anaconda/)), an environment 
can be recreated automatically by running the following command:

```bash
conda create -n schelling --file env_spec.txt
```

Once that is completed, activate the environment by running:

```bash
source activate schelling
```

When you are done with the session, deactivate the environment with:

```bash
source deactivate
```
