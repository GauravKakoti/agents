# Web Agent
The goal of this project is to make a locally runable agent that can autonomously collect data on request from a client.

## Installation instructions:
Once you have cloned this repo, set up a python environment, we recommend using conda to manage it
Note that some of the bulkier packages in requirements.txt are for omniparser

conda create -n webagent
pip install -r  src/requirements.txt

playwright install

## Examples scripts
We recommend looking through webagent_tutorial.ipynb in the notebooks folder to walk through the aspects of the agent. You can run the full stack in src by running agents.py, or use that as an example for your own usage
