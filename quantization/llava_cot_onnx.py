import gc
import os
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Add memory cleanup at the start
if torch.backends.mps.is_available():
    gc.collect()
    torch.mps.empty_cache()
elif torch.cuda.is_available():
    torch.cuda.empty_cache()

# Step 1: Load Model and Processor
model_id = "Xkev/Llama-3.2V-11B-cot"

print("Starting the process...")
start_time = time.time()

print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(model_id)

# Explicitly load model on GPU/MPS
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

model_load_start = time.time()
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
    low_cpu_mem_usage=True,
).to(device)
model_load_end = time.time()
print(f"Model loaded in {model_load_end - model_load_start:.2f} seconds")

# Step 2: Prepare Sample Inputs
print("Preparing sample inputs...")
sample_messages = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "Describe this image"}],
    }
]

# Prepare text and image inputs
sample_text = processor.apply_chat_template(sample_messages, add_generation_prompt=True)
sample_image = Image.open("omniparser_test.png")  # Make sure this image exists

input_prep_start = time.time()
sample_inputs = processor(images=sample_image, text=sample_text, return_tensors="pt")
sample_inputs = {
    k: v.to(device) for k, v in sample_inputs.items()
}  # Move inputs to GPU/MPS
input_prep_end = time.time()
print(f"Inputs prepared in {input_prep_end - input_prep_start:.2f} seconds")

# Step 3: Export to ONNX
print("Exporting model to ONNX...")
os.makedirs("llava-onnx", exist_ok=True)
onnx_path = "llava-onnx/llava_11b_cot_model.onnx"
print(f"Saving to {onnx_path}")
export_start = time.time()
try:
    with torch.no_grad():
        torch.onnx.export(
            model,
            (sample_inputs,),
            onnx_path,
            opset_version=16,
            input_names=list(sample_inputs.keys()),
            output_names=["logits"],
            dynamic_axes={
                # Modified dynamic axes configuration
                k: (
                    {0: "batch_size", 1: "sequence_length"}
                    if any(x in k for x in ["ids", "mask", "labels"])
                    else (
                        {0: "batch_size"}
                        if "pixel_values" in k
                        else {0: "batch_size", 1: "sequence_length"}
                    )
                )
                for k in sample_inputs.keys()
            },
            do_constant_folding=True,
            verbose=False,
        )
    export_end = time.time()
    print(f"Model successfully exported to {onnx_path}")
    print(f"Export completed in {export_end - export_start:.2f} seconds")
except Exception as e:
    print(f"Export failed: {str(e)}")

# Clean up memory after export
del model
del sample_inputs
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
elif torch.cuda.is_available():
    torch.cuda.empty_cache()

total_time = time.time() - start_time
print(f"Total execution time: {total_time:.2f} seconds")
