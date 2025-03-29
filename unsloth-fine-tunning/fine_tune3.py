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

# Environment variables to optimize CUDA memory usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force use of GPU 0

# Choose a much smaller model for 3.8GB GPU
# Option 1: TinyLlama (1.1B parameters)
mlname = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Option 2: Phi-2 (2.7B parameters)
# mlname = "microsoft/phi-2"

# Dataset loading - avoid streaming since we need length
file_path = os.path.join(os.getcwd(), 'mini_data')
ds = load_dataset(
    path=file_path,
    data_files=['occupancy-events-20240301.converted.txt'],
    split="train"
)

# Optional: If dataset is too large, take only a subset
# ds = ds.select(range(min(1000, len(ds))))

# Maximum aggressive quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # Using float16 for memory efficiency
)

# Model loading with CPU offloading for some layers
model = AutoModelForCausalLM.from_pretrained(
    mlname,
    quantization_config=quantization_config,
    device_map="auto",
    max_memory={0: "2GB", "cpu": "16GB"},  # More aggressive CPU offloading
    offload_folder="offload_folder",  # Enable disk offloading if memory still insufficient
)
tokenizer = AutoTokenizer.from_pretrained(mlname)
tokenizer.pad_token = tokenizer.eos_token

# Minimal LoRA configuration to reduce memory footprint
peft_config = LoraConfig(
    r=2,  # Minimal rank
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.0,  # No dropout to save memory
    bias="none",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

def clear_gpu_memory():
    """Clean up GPU memory before training"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU memory cleared. Current allocation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

if __name__ == '__main__':
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    # Training configuration with minimal memory requirements
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=1,  # Minimal batch size
            gradient_accumulation_steps=8,  # Reduced to save memory during accumulation
            gradient_checkpointing_kwargs={"use_reentrant": False},  # More stable checkpointing
            warmup_steps=2,  # Minimal warmup
            max_steps=40,  # Reduced steps
            fp16=True,  # Force fp16
            bf16=False,  # Disable bf16
            logging_steps=1,
            output_dir='./logs',
            optim='adamw_8bit',  # 8-bit optimizer
            learning_rate=5e-5,  # Lower learning rate
            weight_decay=0.0,  # No weight decay to save computation
            lr_scheduler_type='constant',  # Simpler scheduler
            seed=3047,
            # Memory optimizations
            gradient_checkpointing=True,  # Enable gradient checkpointing
            torch_compile=False,  # Disable torch compile
            dataloader_drop_last=True,  # Drop incomplete batches
            dataloader_num_workers=0,  # No parallel data loading
            ddp_find_unused_parameters=False,
            report_to="none",  # Disable reporting
        ),        
    )
    
    # Try/except block to handle potential memory issues gracefully
    try:
        trainer.train()
        # Save the model
        model.save_pretrained(os.path.join(os.getcwd(), 'weight_models_sft'))        
        tokenizer.save_pretrained(os.path.join(os.getcwd(), 'weight_models_sft'))
        print("Training completed successfully!")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("\n\nCUDA OUT OF MEMORY ERROR!")
            print("Suggestions:")
            print("1. Use an even smaller model (like TinyLlama)")
            print("2. Use CPU-only training (remove device_map and set device='cpu')")
            print("3. Try cloud GPU services with more memory")
            print("\nError details:", str(e))
        else:
            raise e