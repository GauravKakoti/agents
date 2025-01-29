# Web Agent library
This is a python-based web agent framework that uses Playwright (currently) and multimodal models to interact with web pages and (eventually) scrape from the web. 

## Requirements

- Python 3.11+
- Playwright installed with browsers
- An API key for accessing a multi-modal LLM (we will assume Groq throughout)
- Some sort of installation of conda

## Installation

1. Clone this repository:
```bash
git clone git@github.com:fellowship/web-agent.git
cd web-agent
```
2. Create a conda environment for this project
```bash
conda create --name fellowship python=3.11
```
3. Install the requirements
```bash
pip install -r src/requirements.txt
```
4. In the terminal export your API key, e.g. 
```bash
export GROQ_API_KEY=<your_api_key>
```
and confirm its been set by
```bash
echo $GROQ_API_KEY
```
download the model weights by running 
```bash
python download_models.py
```
you should see the api key you had set in the first part. 

### Running the agent
Currently in the file `agents.py`, one can run a version of the web agent. To run it please run 
```bash
PYTHONPATH=. python src/agents.py
```
Alternately, you can run the streamlit version of the app with the following command
```bash
streamlit run src/app.py
```
