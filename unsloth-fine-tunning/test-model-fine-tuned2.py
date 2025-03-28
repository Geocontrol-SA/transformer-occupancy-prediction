import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# Correct way to load the tokenizer and model
model_path = os.path.join(os.getcwd(), 'weight_models_sft')


from fine_tune4 import quantization_config

# Load the model
model = AutoModelForCausalLM\
    .from_pretrained(model_path, \
        quantization_config=quantization_config,\
        device_map="cpu",
        max_memory={0: "2GB", "cpu": "16GB"},  # More aggressive CPU offloading
        offload_folder="offload_folder",  
)


# Load the tokenizer from the same path
tokenizer = AutoTokenizer.from_pretrained(model_path)

# from fine_tune3 import model as before_model, tokenizer as before_tokenizer
# model = before_model
# tokenizer = before_tokenizer

# Test the model
# prompt = "Thursday, no rain and warm, route 4391, scheduled at 00:01 and started at 00:02. The occupancy level at bust stop 3637 is:"
prompt = "Thursday, no rain and warm, route 2296, scheduled at 00:01 and started at 00:01. The occupancy level at bust stop 43 is:"
inputs = tokenizer(prompt, return_tensors="pt")

# Move to GPU if available
if torch.cuda.is_available():
    inputs = inputs.to('cuda:0')
    model = model.to('cuda:0')

# Generate prediction
outputs = model.generate(**inputs, max_new_tokens=200)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"result: {result}")
