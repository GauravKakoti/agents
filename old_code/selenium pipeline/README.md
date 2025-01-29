
# Selenium Pipeline

This a pipeline which currently uses uses selenium as an action model. It opens a target url(for testing purposes), takes the screenshot and then passes it to OmniParser. The annotated image goes to an LLM which tells which button to click then web driver then goes to that particular site. 
It can be added into a loop, but this is just a trial, so that everyone has a headstart.


*Please be careful while setting the path to your local machines.*

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
3. Finally install 
```
pip install ollama
pip install selenium
pip install webdriver-manager
```

