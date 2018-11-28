# Paper-reviewer affinity modeling for OpenReview

A key part of matching papers to reviewers is having a good model of paper-reviewer affinity. This repository holds code and tools for generating affinity scores between papers and reviewers.

## Experiment Workflow

This section describes the steps to setup, train, and evaluate an affinity model in this pipeline.

### Configuration
The experimenter starts by creating an experiment directory (e.g. `/exp_1`), and a configuration file inside that directory. Config files should be formatted as JSON and must include the following attributes:

1) `name`: a string that identifies the experiment (avoid using spaces in this field).
2) `dataset`: a string representing the directory where the dataset is located.
3) `model`: a string that specifies the model module to be trained (from `expertise.models`).
4) `keyphrases` a string that specifices the keyphrases module to be used (from `expertise.preprocessors`).

(See `/samples/sample_experiment/config.json` for an example)

All other attributes in the config file are specific to the type of model and experiment being run.

### Setup
To setup a model, run `openreview.expertise.setup_model` with the path to your configuration file as an argument:

```
python -m openreview.expertise.setup_model /samples/sample_experiment/config.json
```

`openreview.expertise.setup_model` imports the model specified in the configuration and passes this configuration and the dataset into the model's `setup` function. `openreview.expertise.setup_model` will then create a directory, `/setup`, in the experiment directory. The model's `setup` function is expected to store files needed for training to the `/setup` directory. The contents of `/setup` are specific to each model.

### Training
writes out all the files that are needed to run Model.train() and Model.evaluate().
The contents of /setup are specific to each model. models should know how to use them.

```
python -m openreview.expertise.train_model /samples/sample_experiment/config.json
```




