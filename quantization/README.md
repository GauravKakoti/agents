This directory is for all scripts related to generating quantized versions of various models. It is put into a separate folder since to run quantization requires a carefully created environment. The main files in this directory are the notebook `llava_cot_quantization.ipynb` and `llava_cot_onnx.py`. The former deals with creating quantized versions of models and the latter with exporting the model to the ONNX format. 

## environment
The main libraries for running quantization are `transformers` and `bitsandbytes`, however there are some dependencies on specific versions of `scipy` that are required. The file `quantization_requirements.txt` were generated through `pip freeze` in the environment I used. The main libraries to pay attention to are `transformers`, `torch`, `bitsandbytes`, `scipy`, and `numpy`. 

Secondly, these libraries only work for a Linux or Windows backend (sorry Apple Silicon users). When creating these quantized models and exporting to ONNX, a GPU is necessary. In my case I created a GPU instance with a single L4 GPU with 24 GB of vRAM. This seemed to work fine in my case. For ONNX conversion, you will also need to pip install the libraries `onnx` and `onnxruntime` (the latter for inference). 


