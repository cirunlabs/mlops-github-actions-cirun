# GitHub Actions with your own GPUs

## Steps to have this repository working

## Installation
Install python for your OS

[Installation for your OS](https://www.python.org/downloads/)

## After installation of python 
 - Clone this repository. 
 - Open cloned repository in code editor. 

## Configuration of Requirements
```
pip install -r requirements.txt
```
## For the training of model 
```
python3 train.py
```
# Setup of GitHub Actions to train ML models with own GPUs

## Prerequisites

Have fully working ML model along with it's resources.

If data is too big for GitHub, can use DVC- [Data Version Control](https://dvc.org/doc)

## Setup of GitHub action

- Create a Workflow.
- We need to configure CML- [Continous machine learning](https://github.com/iterative/cml#getting-started) with repository to setup the enviornment for training.
- CML come preinstalled in Docker Images.
- To setup CML with GitHub actions use docker container.
```
docker://dvcorg/cml-py3:latest
```
- We don't need any other container argument for CPU.
- To setup CML with GPUs use container.
```
ghcr.io/iterative/cml:0-dvc2-base1-gpu
```
- Also needs a container argument to configure container for all GPUs.
```
--gpus all
```
- We also have to pass our repository's GITHUB_TOKEN so, that authentication is done on behalf of GitHub Actions. 
- Then we have steps what ever we want to do with our Workflow.

## To Reproduce the example's Workflow

- Create a Workflow.
- On every push trigger the workflow.
```
on:
 push:
   branches: [ "new-example-2" ] # When a push occurs on a branch workflow will trigger.
 workflow_dispatch:
```

- In example we triggers two jobs to train on our Model on CPU and GPU.
- For this we are using Matrix for jobs.
- Creating a 2d matrix to execute our jobs with respective docker containers.
```
jobs:
 CPU_GPU_matrix: # Creating matrix for jobs.
   strategy:
     matrix:
       os: [ubuntu-latest, self-hosted]
       containers: ["docker://dvcorg/cml-py3:latest" , "ghcr.io/iterative/cml:0-dvc2-base1-gpu" ]
       container_arg: [" " , "--gpus all"]
```

- To avoid the unwanted combinations we are using exclude property.
- Excluding the unwanted docker container and container_arg in our respective OS.
```
exclude:
         - os: ubuntu-latest
           containers: "ghcr.io/iterative/cml:0-dvc2-base1-gpu"
         - os: ubuntu-latest
           container_arg: "--gpus all"
         - os: self-hosted
           containers: docker://dvcorg/cml-py3:latest
         - os: self-hosted
           container_arg: " "
```

- Dynamically passing of os, containers, container_agr using matrix
```
runs-on: ${{ matrix.os }}
   container:
     image: ${{ matrix.containers }}
     options: ${{ matrix.container_arg }}
```

- Then have authentication and steps we want.
```
steps:
     - uses: actions/checkout@v3
     - name: Get Free Memory
       run: free -h
     - name: Dependency Install
       run: pip install -r requirements.txt
     - name: MLops
       env:
         repo_token: ${{ secrets.GITHUB_TOKEN }}
       run: |
         # Your ML workflow goes here
         python train.py
```

# Running Actions on your GPUs.
- To have GPUs on your Actions, we have create a ```.cirun.yml``` configuration file.
- In example we used AWS self-hosted runner with ```Tesla T4``` GPU.
```
# Self-Hosted Github Action Runners on AWS via Cirun.io
runners:
 - name: gpu-runner
   # Cloud Provider: AWS
   cloud: aws
   # VM on AWS
   instance_type: g4dn.xlarge
   # Ubuntu, ami image, region
   region: eu-west-1
   machine_image: ami-00ac0c28c01352e53
   # Add this label in the "runs-on" param in .github/workflows/<workflow-name>.yml
   # So that this runner is created for running the workflow
   workflow: .github/workflows/MLops.yml
   labels:
     - cirun.gpu
```










