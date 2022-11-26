# GitHub Actions with your own GPUs

## Quickstart

To setup MLOps using Cirun, refer to this **[example](https://cirun-examples--cirun-docs.netlify.app/examples/actions_with_gpu)**.
## Setting up Actions for MLOps using Cirun

### Prerequisites
Have fully working ML model along with it's resources.

If data is too big for GitHub, can use DVC- [Data Version Control](https://dvc.org/)

### Setup of GitHub action

The workflow file used in this project: .github/workflows/MLOps.yaml 

- Create a Workflow.
- Set your working branch.
```yml
 push:
    branches: [ "none" ] #  When a push occurs on a branch workflow will trigger on your desired branch.
```
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
- In example we used AWS self-hosted runner with ```Tesla T4``` GPU.

## References
- [GitHub Actions](https://docs.github.com/en/actions)
- [MLOps](https://ml-ops.org/)
- [CML](https://cml.dev/)
- [Cirun](https://docs.cirun.io/)
- [MLOps using Cirun](https://cirun-examples--cirun-docs.netlify.app/examples/actions_with_gpu)