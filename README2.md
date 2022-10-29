# Asymmetric Dense Passage Retrieval - AsymDPR
This repo is the project to evaluate/analyse the effect of asymmetric dual encoder in dense passage retrieval

# Environment Setup
Ideally setup a conda environment and install all the requirements. `jobfiles` folder consist of all the required .sh files to run on Lisa. 
Use `lisasetup.job` to setup your environment.

# Downloading Dataset and Setting up Project
To download the required dataset, use the following jobfiles in your lisa environment

```
sbatch jobs/downloaddatajob.sh
```
To prepare the subset of data for reproducing the results, run 
```
sbatch jobs/transformdata.sh
```

## Training Pipeline
To run the baseline experiment, directly trigger the following job file

```
sbatch jobs/trainDPRBaseline.job
```

Results are saved in ```outputs/<model_name>``` directory. All evaluation results can be found inside.

### Reproducing table 1 (Combination of Query, Passage encoders)

All files inside jobs/ folder take care of running the expierment. Run ```jobs/trainDPR<query_encodername>``` to run training pipeline for the different query encoder-passage encoder setting


### Generating Index

To evaluate the model performance, use ```sbatch jobs/generateIndex.job``` to use the DPR model to index the passages from the train+eval set created in the first step


### Ablation Experiment

To visualize the ablation experiment, run ``` sbatch jobs/bertbaseabltaion.sh``` to run the DPR with bert-base as query, passage encoder and removing tranformer layers from the model.

