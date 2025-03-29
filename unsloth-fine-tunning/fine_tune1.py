from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os

max_seq_length_param = 4096
# mlname = 'unsloth/llama-3-8b-bnb-4bit'  # Teste também com "meta-llama/Llama-3-8B"
mlname = "unsloth/mistral-7b-v0.3-bnb-4bit"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Força uso da GPU 0


file_path = os.path.join(os.getcwd(), 'data')
ds = load_dataset(
    path=file_path,
    data_files=['vehicle_bus_stop_occupation_20250101_20250201-copy.speaks.txt'],
    split="train"
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Carrega o modelo SEM FastLlamaModel
model = AutoModelForCausalLM.from_pretrained(
    mlname,
    quantization_config=quantization_config,
    device_map="auto",
    max_memory={0: "5GB"},  # Ajuste crítico
)
tokenizer = AutoTokenizer.from_pretrained(mlname)
tokenizer.pad_token = tokenizer.eos_token  # Ajuste crítico

# Preparação PEFT
peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)


if __name__ == '__main__':
    # Configuração do Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,    
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=60,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            output_dir='./logs',
            optim='adamw_8bit',
            weight_decay=0.01,
            lr_scheduler_type='linear',
            seed=3047
        ),
    )

    trainer.train()
    # como foi usado o método get_peft_model, ele sobrescreve o método save_model
    # trainer.save_model(output_dir=os.path.join(os.getcwd(), 'weight_models'))

    model.save_pretrained(os.path.join(os.getcwd(), 'weight_models_sft'))
    tokenizer.save_pretrained(os.path.join(os.getcwd(), 'weight_models_sft'))


