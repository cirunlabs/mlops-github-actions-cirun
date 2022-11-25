# GitHub Actions with your own GPUs

## Quickstart

To get started, clone this repository and follow these steps.

- Install [python](https://www.python.org/downloads/) for your OS

- Installing Requirements
```yml
pip install -r requirements.txt
```
- For training of model
```yml
python3 train.py
```
## Setting up Actions for MLOps using Cirun

### Prerequisites
Have fully working ML model along with it's resources.

If data is too big for GitHub, can use DVC- [Data Version Control](https://dvc.org/)

### Setup of GitHub action

The workflow file used in this project: .github/workflows/MLOps.yaml 

- Create a Workflow.
- We need to configure CML- [Continuous Machine Learning](https://github.com/iterative/cml#getting-started)
- CML Docker container comes with pre-installed dependencies which are important for full-stack data science.
- To setup CML with GitHub actions use docker container.
```yml
docker://dvcorg/cml-py3:latest
```
- To setup CML with GPUs use container.
```yml
ghcr.io/iterative/cml:0-dvc2-base1-gpu
```
- Followed by Container argument
```yml
--gpus all
```
- We must have to pass our repository's GITHUB_TOKEN so, that authentication is done on behalf of GitHub Actions.
```yml
env:
          repo_token: ${{ secrets.GITHUB_TOKEN }}
```
- Then we have steps what ever we want to do with our Workflow.

### Running Actions on your GPUs using Cirun.

- To have GPUs on your Actions, we have create a ```.cirun.yml``` configuration file.
- In project we used AWS self-hosted runner with ```Tesla T4``` GPU.
