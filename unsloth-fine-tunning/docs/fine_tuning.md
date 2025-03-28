## How to make a Fine tune

There are to two aprroaches todo this, one where you can use a fine_tune.py a another using a python notebooks presents in this project. This documentation is if you wants to use the first way. For other way, come back to root documentation, and choice in options bellow, between: Mistral, Tinny-llama and Phi-2.

### Data Preparation

[docs - data_prepare](/docs/data_prepare.md)

### Running without python notebooks

#### Check datasets if you're using jsonl files

```bash
python fine_tune.py --mode test_dataset_txt --datasets_path ./data

```

### Check datasets if you're using text files

```bash
python fine_tune.py --mode test_dataset_txt --datasets_path ./data

```

### Convert datasets if you're using json files

```bash
python fine_tune.py --mode convert_dataset --datasets_path ./data

```

### Convert datasets if you're using text files

```bash
python fine_tune.py --mode convert_dataset_txt --datasets_path ./data

```

### Running fine tune with mistral example

```bash
python fine_tune.py --mode fine_tune --datasets_path ./data --model_name 'unsloth/mistral-7b-v0.3-bnb-4bit'
```

### Running prediction with mistral example

```bash
python fine_tune.py --mode predict --datasets_path ./data --model_name 'unsloth/mistral-7b-v0.3-bnb-4bit'
```

### For cli commands questions

```bash
python fine_tune.py -h
```
