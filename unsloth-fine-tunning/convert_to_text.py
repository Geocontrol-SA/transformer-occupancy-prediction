import pandas as pd
from unsloth import FastLanguageModel

import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset


def list_devices():
    # Verifica se CUDA (GPU) está disponível
    if torch.cuda.is_available():
        print("CUDA (GPU) está disponível.")
        # Lista todos os dispositivos GPU disponíveis
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA (GPU) não está disponível. Usando CPU.")

    # Lista o dispositivo padrão
    print(f"Dispositivo padrão: {torch.cuda.current_device()}")

list_devices()


# # mlname = 'unsloth/Qwen2.5-0.5B-bnb-4bit'
mlname = 'unsloth/tinyllama-bnb-4bit'
# mlname = 'microsoft/phi-2'

max_seq_length=2048 #2048 #1024
# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/tinyllama-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=None,  # None for auto detection, or torch.float16, or torch.bfloat16
    load_in_4bit=True
)

# Prepare model for inference
model = FastLanguageModel.for_inference(model)
print(f"model.device: {model.device}")





def generate_text(text: str):
    inputs = tokenizer(text, return_tensors="pt").to('cuda:0')
    print(f"inputs: {inputs}")
    outputs = model.generate(**inputs, max_new_tokens=20)
    # print(f"outputs: {outputs}")
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"result: {result}")
    return result



peft_model = FastLanguageModel.get_peft_model(
    model=model, r=16, 
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_alpha=16,
    lora_dropout=0,
    bias='none',
    use_gradient_checkpointing=True,
    random_state=3407,
    max_seq_length=max_seq_length,
    use_rslora=False,
    loftq_config=None
)




    





if __name__ == '__main__':
    # Load the CSV file into a pandas DataFrame
    # df = pd.read_csv('data/vehicle_bus_stop_occupation_20250101_20250201-copy.csv')
    # print(df.columns)
    # print(df.dtypes)

    # print(df.head())
    # print(df[:10])

    generate_text("<human>Voce é qual inteligencia artificial?\n<bot>:")

