# Bus Occupancy Prediction

    UFES - Tópicos Especiais em Informática (GPT)

> _Predicting the occupancy rate of public transportation buses at bus stops using transformer-based models._

The main goal of this project is to predict the occupancy rate of public transportation buses at bus stops by using transformer-based models. The model will integrate various features extracted from the trip data, including but not limited to: Temporal Data, Route and Vehicle Information, Passenger Metrics, Operational Details, Service Disruptions, and Stop-Level Data. By leveraging these inputs, the model aims to forecast occupancy levels with high accuracy, thereby facilitating improved operational decision-making and enhanced service planning.

### Approaches

As described in the [report](report/Trabalho_GPT_Eduardo_Bruno_Igor.pdf), the project explores two main approaches for occupancy prediction:

- **Pytorch Model**: A custom transformer-based model is developed using PyTorch. This model is designed to handle the complexities of the data, including variable-length sequences and class imbalance, while providing real-time occupancy predictions for each bus stop in a trip.
- **Unsloth Fine-tuning**: The second approach involves fine-tuning a pre-trained transformer model using the Unsloth framework, which provides a robust environment for training and evaluating transformer models on text data. This method aims to leverage existing knowledge from large-scale datasets to improve the performance of occupancy predictions by converting structured transportation records into a textual format.

### Project Structure

##### 1. Pytorch Model

The pytorch model is located in the [pytorch-model](pytorch-model) folder. The model is trained using the `train.py` script.
More information about the model can be found in the [README](pytorch-model/README.md) file.

##### 2. Unsloth Fine-tuning

The unsloth finetuning is located in the [unsloth-fine-tunning](unsloth-fine-tunning) folder.
More information about it can be found in the [README](unsloth-fine-tunning/README.md) file.

##### 3. Report

The [report](report/Trabalho_GPT_Eduardo_Bruno_Igor.pdf) is located in the report folder.
The report contains a detailed description of the project, including the methodology, results, and conclusions drawn from the experiments conducted.

### Authors

| Name                  | Email                                                             |
| --------------------- | ----------------------------------------------------------------- |
| Bruno Rocha Toffoli   | [btoffoli@gmail.com](mailto:btoffoli@gmail.com)                   |
| Eduardo Lima Pereira  | [eduardo.pereira@ifes.edu.br](mailto:eduardo.pereira@ifes.edu.br) |
| Igor Aguiar Rodrigues | [igor_aguiar@yahoo.com.br](mailto:igor_aguiar@yahoo.com.br)       |
