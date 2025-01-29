
# OmniParser + LLM

This a part of the pipeline where we will take the screenshot of the current web page and then pass it to an LLM model, the third stage of the pipeline , i.e. using LaVAgue to navigate is still a work in progress


Please be careful while setting the path to your local machines.

## Installation

1. Clone OmniParser using the following command and install the requirements for the same

```bash
  git clone https://huggingface.co/microsoft/OmniParser
  pip install -r requirements.txt
  ```

2. Download Ollama and the model of your choice(I used llama3.2) if you want it to run on your system locally:
```
    https://ollama.com/download/windows
```

else you can import from langchain as follows:
```
from langchain_community.llms import Ollama
llm = Ollama(model = 'llama3.2')
result = llm.invoke('INSERT_YOUR_PROMPT_HERE')
print(result) 
```
3. Finally install and import Ollama
```
pip install ollama
```
```
import ollama
```
