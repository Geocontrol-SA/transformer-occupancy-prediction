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
from dotenv import load_dotenv
import argparse
import logging
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict

# from convert_data_to_json import convert_to_dict, convert_to_text

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Força uso da GPU 0



# days_of_week = {
#     0: "domingo",
#     1: "segunda-feira",
#     2: "terça-feira",
#     3: "quarta-feira",
#     4: "quinta-feira",
#     5: "sexta-feira",
#     6: "sabado"
# }

days_of_week = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday"
}

TZ=timezone(timedelta(hours=-3))

def convert_tz(str_dt: str, tz: timezone):
    d = datetime.fromisoformat(str_dt)
    return d.astimezone(tz)

header = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

{"instruction":"","input":"wednesday, no rain and cold, route 2856, scheduled at 00:00 and started at 00:00. The occupancy level at bust stop 3241 is:","output":"0"}

# human_speaks = "Sabendo que está %clima_tempo e a temperatura é %temperatura. Qual a lotação da linha %linha para o ponto %ponto para %dia às %hora?"
# human_speaks = "Sendo o dia da semana %day_of_week, com %weather_precipitation e %weather_temperature, para uma viagem do itinerario %trip_route_id, que geralmente inicia %trip_scheduled_time e que iniciou %trip_start_time e terminou %trip_end_time. Qual o nivel de ocupacao para o ponto %bus_stop_id?"
# delayed_human_speaks = "Para uma viagem do itinerario %trip_route_id, que geralmente começa %trip_scheduled_time mas que começou %trip_start_time e terminou %trip_end_time, no dia %days_of_week, %weather_precipitation e temperatura %weather_temperature. Qual o nível de lotação para o ponto %bus_stop_id?"

human_speaks="%day_of_week, %weather_precipitation and %weather_temperature, route %trip_route_id, scheduled at %trip_scheduled_time and started at %trip_start_time. The occupancy level at bust stop %bus_stop_id is:"



# bot_speaks = "Sendo o dia da semana %day_of_week, no ponto %busStopId, o nivel de ocupacao é %occupancyLevel, para o itinerário %trip_route_id"

# bot_speaks = "Sendo o dia da semana %day_of_week, no ponto %busStopId, o nivel de ocupacao é %occupancyLevel, para o itinerário %trip_route_id"

bot_speaks = "%occupancyLevel"

def convert_to_text(
        register: dict,
    ):
    resp = convert_to_json(register)
    return resp['text']
    # return f"<human>{resp['input']}<bot>{resp['']}"
    # return f"""<human>: {resp['input']}\n<bot>: {resp['output']}\n\n"""


    

def convert_to_json(
        register: dict,
    ):
    logger.debug(f"Register(type): {type(register)}")
    logger.debug(f"Register: {register}")

    trip_route_id = register['tripRouteId']
    trip_scheduled_time = register['tripScheduledTime']
    trip_start_time = register['tripStartTime']
    trip_end_time = register['tripEndTime']        
    weather_precipitation = register['weatherPrecipitation']
    weather_temperature = register['weatherTemperature']
    bus_stop_id = register['busStopId']
    occupancy_level = register['occupancyLevel']
    normalized_location = register['busStopLocation']/register['routeTotalLength'],  


    scheduled_datetime = convert_tz(trip_scheduled_time, TZ)
    scheduled_time = scheduled_datetime.strftime("%H:%M")
    start_datetime = convert_tz(trip_start_time, TZ).strftime("%H:%M")
    end_time = convert_tz(trip_end_time, TZ).strftime("%H:%M")
    
    dw = days_of_week[scheduled_datetime.weekday()]

    # occupancy = 'Lotado' if occupancy_level > 1 else 'Vazio' if occupancy_level < 1 else 'Cheio'

    occupancy = str(occupancy_level)



    human_speaking = human_speaks\
        .replace("%trip_route_id", str(trip_route_id))\
        .replace("%trip_scheduled_time", scheduled_time)\
        .replace("%trip_start_time", start_datetime)\
        .replace("%trip_end_time", end_time)\
        .replace("%day_of_week", dw)\
        .replace("%weather_precipitation", 'heavy rain' if weather_precipitation > 2 else 'light rain' if weather_precipitation > 0 else 'no rain')\
        .replace("%weather_temperature", 'hot' if weather_temperature > 27 else 'warm' if weather_temperature > 18 else 'cold')\
        .replace("%bus_stop_id", str(bus_stop_id))

    bot_speaking = bot_speaks\
        .replace("%busStopId", str(bus_stop_id))\
        .replace("%trip_route_id", str(trip_route_id))\
        .replace("%day_of_week", dw)\
        .replace("%occupancyLevel", occupancy)
    

    text = f"{header}\n\n### Instruction:\n{human_speaking}\n\n### Response:\n{bot_speaking}\n\n\n"
    

    return {
        "instruction": human_speaking,
        "input": "",
        "output": bot_speaking,
        "text": text,               
    }
    # return f"""<human>: {human_speaking}\n<bot>:{bot_speaking}\n\n"""



    

class BusOccupancyFineTune:
    """
    This is a fine tune class to manage the fine tune process for bus occupancy data.
    """

   
    def __init__(self, mode: str = "test_dataset",  **kwargs):
        
        self.model_name = kwargs.get("model_name", "unsloth/mistral-7b-v0.3-bnb-4bit")
        output_default_dir = f'tuned_{self.model_name}'
        self.datasets_path = kwargs.get("datasets_path", "./data")
        self.output_dir = kwargs.get("output_dir", output_default_dir)
        self.data_batch_size = kwargs.get("data_batch_size", 1000)  # New parameter for data batch size

        self.preprocess = kwargs.get("preprocess_data", False)
        
        self.__load_training_args()

        
        if mode in ['predict', 'fine_tune']:
            self.__load_config()
        
        if mode == 'predict':
            self.__load_fine_tuned_model()

        elif mode == 'fine_tune':
            self.__load_dataset()
            self.__load_model_for_training()
            
            
    
    def train(self):
        if self.data_batch_size:
            for batch in self.train_dataset:                
                self.trainer = SFTTrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    args=self.training_args,
                    train_dataset=batch
                )
                self.trainer.train()
        else:
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                args=self.training_args,
                train_dataset=self.train_dataset,
            )
            self.trainer.train()

        

        self.model.save_pretrained(os.path.join(os.getcwd(), self.output_dir))
        self.tokenizer.save_pretrained(os.path.join(os.getcwd(), self.output_dir))
        
    def __load_config(self):        
            if self.model_name == 'microsoft/phi-2':
                self.quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    lora_dropout=0.0,  # No dropout to save memory
                    bias="none",
                    bnb_4bit_compute_dtype=torch.float16,  # Using float16 for memory efficiency
                )

                self.peft_config = LoraConfig(
                    r=2,  # Minimal rank
                    lora_alpha=16,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_dropout=0.0,  # No dropout to save memory
                    bias="none",
                )            
            elif self.model_name == 'unsloth/mistral-7b-v0.3-bnb-4bit':
                self.quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                            # Preparação PEFT
                self.peft_config = LoraConfig(
                    r=8,
                    lora_alpha=8,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_dropout=0.1,
                    bias="none",
                )
            elif self.model_name == 'TinyLlama/TinyLlama-1.1B-Chat-v1.0':
                self.quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    max_memory={0: "2GB", "cpu": "16GB"},  # More aggressive CPU offloading
                    bnb_4bit_compute_dtype=torch.float16,  # Using float16 for memory efficiency
                )

                self.peft_config = LoraConfig(
                    r=2,  # Minimal rank
                    lora_alpha=16,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_dropout=0.0,  # No dropout to save memory
                    bias="none",
                )
                        
            else:
                logger.warning("Unknown model name supported...")


    
                
            

           
        

    def run(self):
        self.train()

    def __load_training_args(self):
        if self.model_name == 'microsoft/phi-2':
            self.training_args = TrainingArguments(
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
            )

            
        elif self.model_name == 'unsloth/mistral-7b-v0.3-bnb-4bit':
            self.training_args = TrainingArguments(
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
            )

        elif self.model_name == 'TinyLlama/TinyLlama-1.1B-Chat-v1.0':
            self.training_args = TrainingArguments(
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
            )
        else:
            logger.warning("Unknown model name supported...")
            self.training_args = None

    def __load_dataset(self):
        if self.data_batch_size:
            ds = self.read_datasets_in_batches()        
        else:
            ds = load_dataset(
                path=self.datasets_path,
                data_files=sorted(os.listdir(self.datasets_path)),
                split="train",             
            )
            if self.preprocess:
                ds = ds.map(self.__preprocess_data, batched=True)

        self.train_dataset = ds
            

    def __load_model_for_training(self, **kwargs):
        self.max_length = kwargs.get("max_length", 128)
        self.batch_size = kwargs.get("batch_size", 128)
        self.learning_rate = kwargs.get("learning_rate", 2e-5)
        self.num_epochs = kwargs.get("num_epochs", 3) 
        load_in_4bit = kwargs.get("load_in_4bit", False)
        bnb_4bit_compute_dtype = kwargs.get("bnb_4bit_compute_dtype", "float16")
        bnb_4bit_quant_type = kwargs.get("bnb_4bit_quant_type", "nf4")
        bnb_4bit_use_double_quant = kwargs.get("bnb_4bit_use_double_quant", True)

        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=load_in_4bit,
        #     bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        #     bnb_4bit_quant_type=bnb_4bit_quant_type,
        #     bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        # )

        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     llm_int8_enable_fp32_cpu_offload=True
        # )

        # quantization_config = BitsAndBytesConfig(
        #     load_in_8bit=True,
        #     llm_int8_enable_fp32_cpu_offload=True
        # )

        device_map = {
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.1": 0,
            # ... Add more layers to GPU (device 0) as your memory allows
            "model.layers.2": "cpu",
            "model.layers.3": "cpu",
            # ... Put remaining layers on CPU
            "model.norm": 0,
            "lm_head": 0
        }
        device_map = "auto"

        logger.info(f"self.model_name: {self.model_name}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.quantization_config,
            device_map=device_map,            
            # max_memory=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            # max_memory={0: "2GB"},  # Critical adjustment
            # max_memory={"cuda: 0": "3GB", "cpu": "16GB"},  # More aggressive CPU offloading
            offload_folder=True,  # Enable disk offloading if memory still insufficient
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # peft_config = LoraConfig(
        #     r=8,
        #     lora_alpha=8,
        #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        #     lora_dropout=0.1,
        #     bias="none",
        # )

        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()

        

     


    def __load_fine_tuned_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.output_dir).to("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.output_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def convert_datasets(self, type_of_database: str | Dict = Dict):
        logger.debug(f"Converting datasets: {self.datasets_path}")
        file_path_current = None
        acc = []
        for batch, file_path in self.read_datasets_in_batches(type_of_database):
            if not file_path_current:
                file_path_current = file_path

            # if file_path has changed
            if file_path != file_path_current:
                # store acc in file
                logger.debug(f"Storing acc in file: {file_path_current}")
                if type_of_database == Dict:
                    df_acc = pd.DataFrame(acc)
                    df_acc.to_json(file_path_current.replace('.jsonl', '.converted.jsonl'), orient='records', lines=True)
                else:
                    with open(file_path_current.replace('.jsonl', '.converted.txt'), "w") as f:
                        f.writelines(acc)

                file_path_current = file_path
                acc.clear()

            acc.extend(batch)

        # Handle the last file after the loop
        if file_path_current:
            logger.debug(f"Storing acc in file: {file_path_current}")
            if type_of_database == Dict:
                df_acc = pd.DataFrame(acc)
                df_acc.to_json(file_path_current.replace('.jsonl', '.converted.jsonl'), orient='records', lines=True)
            else:
                with open(file_path_current.replace('.jsonl', '.converted.txt'), "w") as f:
                    f.writelines(acc)


    def read_datasets_in_batches(self, type_of_database: str | Dict = Dict):
        logger.debug(f"Reading datasets in batches: {self.datasets_path}")
        filenames = sorted(os.listdir(self.datasets_path))
        logger.debug(f"filenames: {filenames}")
        acc = []
        for f in filenames:
            if f.endswith('.jsonl'):
                logger.debug(f"Processing file: {f}")
                file_path = os.path.join(self.datasets_path, f)
                if os.path.isfile(file_path):
                    df = pd.read_json(file_path, lines=True)                    
                    for index, row in df.iterrows():
                        if type_of_database == Dict:
                            acc.append(convert_to_json(row))
                        else:
                            acc.append(convert_to_text(row))
                        if len(acc) == self.data_batch_size:
                            yield acc, file_path
                            acc = []
        if len(acc):
            yield acc, file_path
                    
    
    @staticmethod
    def __preprocess_data(tokenizer, ds):
        """Tokenize and preprocess the dataset"""
        inputs = tokenizer(ds['text'], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].long()  # Ensure input_ids are of type Long
        inputs['attention_mask'] = inputs['attention_mask'].long()  # Ensure attention_mask are of type Long
        return inputs
        

if __name__ == '__main__':
    load_dotenv(override=True)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', type=str, default=os.getenv('MODE', 'test_dataset'), choices=['fine_tune', 'predict', 'test_dataset', 'test_dataset_txt', 'convert_dataset', 'convert_dataset_text'])
    argparser.add_argument('--model_name', type=str, default=os.getenv('MODEL_NAME', 'unsloth/mistral-7b-v0.3-bnb-4bit'), 
                           choices=[
                               'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                               'unsloth/mistral-7b-v0.3-bnb-4bit',
                               'microsoft/phi-2',                               
                            ])
    argparser.add_argument('--datasets_path', type=str, default=os.getenv('DATASETS_PATH', './data'))
    argparser.add_argument('--max_length', type=int, default=int(os.getenv('MAX_LENGTH', 4096)))
    argparser.add_argument('--batch_size', type=int, default=int(os.getenv('BATCH_SIZE', 2)))
    argparser.add_argument('--learning_rate', type=float, default=float(os.getenv('LEARNING_RATE', 2e-5)))
    argparser.add_argument('--num_epochs', type=int, default=int(os.getenv('NUM_EPOCHS', 3)))
    argparser.add_argument('--output_dir', type=str, default=os.getenv('OUTPUT_DIR'))
    argparser.add_argument('--load_in_4bit', type=bool, default=os.getenv('LOAD_IN_4BIT', True))
    argparser.add_argument('--bnb_4bit_compute_dtype', type=str, default=os.getenv('BNB_4BIT_COMPUTE_DTYPE', 'float16'))
    argparser.add_argument('--bnb_4bit_quant_type', type=str, default=os.getenv('BNB_4BIT_QUANT_TYPE', 'nf4'))
    argparser.add_argument('--bnb_4bit_use_double_quant', type=bool, default=os.getenv('BNB_4BIT_USE_DOUBLE_QUANT', True))
    argparser.add_argument('--data_batch_size', type=int, default=int(os.getenv('DATA_BATCH_SIZE', 0)))  # New argument
    argparser.add_argument('--preprocessing_data', type=bool, default=os.getenv('PREPROCESSING_DATA', False))

    args = argparser.parse_args()

    logger.info(f"Arguments: {args}")
    

    if args.mode == 'test_dataset':
        bus_occupancy_fine_tune = BusOccupancyFineTune(
            preprocessing_data=args.preprocessing_data,
            datasets_path=args.datasets_path,
            data_batch_size=args.data_batch_size if args.data_batch_size else 10
            )
        # logger.debug(f"Reading datasets in batches: {datasets_path}")
        l = bus_occupancy_fine_tune.read_datasets_in_batches()
        resp = next(l)
        logger.debug(f"Response: {len(resp)}")
        logger.debug('\n'.join([str(i) for i in resp[0]]))
        # logger.debug(resp[0][1])
        l.close()
    
    if args.mode == 'test_dataset_txt':
        bus_occupancy_fine_tune = BusOccupancyFineTune(
            preprocessing_data=args.preprocessing_data,
            datasets_path=args.datasets_path,
            data_batch_size=args.data_batch_size if args.data_batch_size else 10
            )
        # logger.debug(f"Reading datasets in batches: {datasets_path}")
        l = bus_occupancy_fine_tune.read_datasets_in_batches(type_of_database=str)
        resp = next(l)
        logger.debug(f"Response: {len(resp)}")
        logger.debug('\n'.join([str(i) for i in resp[0]]))
        # logger.debug(resp[0])
        l.close()

    if args.mode == 'convert_dataset':
        bus_occupancy_fine_tune = BusOccupancyFineTune(
            preprocessing_data=args.preprocessing_data,
            datasets_path=args.datasets_path,
            data_batch_size=args.data_batch_size if args.data_batch_size else 1000
            )
        bus_occupancy_fine_tune.convert_datasets()
        
    if args.mode == 'convert_dataset_text':
        bus_occupancy_fine_tune = BusOccupancyFineTune(
            datasets_path=args.datasets_path,
            data_batch_size=args.data_batch_size if args.data_batch_size else 1000
            )
        bus_occupancy_fine_tune.convert_datasets(type_of_database=str)


    if args.mode == 'fine_tune':
        bus_occupancy_fine_tune = BusOccupancyFineTune(
            mode='fine_tune',
            model_name=args.model_name,            
            max_length=args.max_length,            
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            output_dir=args.output_dir if args.output_dir else f'tuned_{args.model_name}',
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            batch_size=args.batch_size,
            datasets_path=args.datasets_path,
            data_batch_size=args.data_batch_size  # Pass the new argument
        )
        bus_occupancy_fine_tune.run()
    elif args.mode == 'predict':
        bus_occupancy_fine_tune = BusOccupancyFineTune(mode='predict', model_name=args.model_name)        
        text = input("Enter your text: ")
        logger.debug(bus_occupancy_fine_tune.predict(text))
