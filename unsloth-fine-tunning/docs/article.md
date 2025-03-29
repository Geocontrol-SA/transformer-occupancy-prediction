## Methodology: Fine-tuning Language Models for Bus Occupancy Prediction

### Problem Formulation

In this research, we address the challenge of predicting bus occupancy levels categorized into three discrete states: empty (0), full (1), and overcrowded (2). We explored the potential of transformer-based language models by fine-tuning them using an instruction-based format that transforms transportation data into a structured prediction task.

### Data Preparation and Formatting

The data preparation process underwent a significant transformation to better align with modern large language model (LLM) fine-tuning approaches. We adopted an instruction-based format that provides clear context and a specific prediction request. The new format includes:

```
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
[Day], [precipitation status] and [temperature status], route [route number],
scheduled at [scheduled time] and started at [actual start time].
The occupancy level at bust stop [stop number] is:

### Response:
[Occupancy Level: 0, 1, or 2]
```

Example input:

```
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
Wednesday, no rain and warm, route 536, scheduled at 23:59 and started at 23:59.
The occupancy level at bust stop 4352 is:

### Response:
0
```

### Model Selection and Configuration

We investigated three distinct language models to assess their performance in this specialized prediction task:

1. **Mistral-7b**: A 7 billion parameter model that demonstrated a tendency to respond with JSON-formatted outputs
2. **Microsoft Phi-2**: Another transformer model with unique response characteristics
3. **TinyLlama-1.1B-Chat-v1.0**: A compact model with approximately 1.1 billion parameters

### Model Characteristics

#### Mistral-7b

Mistral-7b is a powerful open-source language model with notable characteristics:

- 7 billion parameters, providing substantial computational capacity
- Known for high performance across various natural language processing tasks
- Developed with an emphasis on efficiency and versatility
- Demonstrates strong few-shot learning capabilities
- Exhibits a tendency to generate structured outputs, particularly JSON

Example output for Mistral-7b:

```
### Instruction:
Thursday, no rain and warm, route 4206, scheduled at 00:20 and started at 00:17.
The occupancy level at bust stop 678 is:

### Response:
{
  "route": "4206",
  "start_time": "00:17",
  "occupancy": "100%"
}
```

Unique Observations for Mistral-7b:

- Consistently generated structured JSON responses
- Included additional metadata beyond the requested occupancy level
- Showed a strong propensity for creating comprehensive, formatted outputs
- Demonstrated more consistent response formatting compared to other models

#### Microsoft Phi-2

Phi-2 is a compact language model developed by Microsoft, characterized by its relatively small size (2.7B parameters) but impressive performance. Key features include:

- Trained on a carefully curated dataset of educational and high-quality texts
- Demonstrates strong reasoning capabilities despite its smaller size
- Designed to be more interpretable and efficient compared to larger models
- Excels in tasks requiring logical reasoning and code understanding

Example output for Phi-2:

```
### Instruction:
Thursday, no rain and warm, route 4206, scheduled at 00:20 and started at 00:17.
The occupancy level at bust stop 678 is:

### Response:
{
  "route": "4206",
  "start_time": "00:17",
  "occupancy": "100%"
}
```

#### TinyLlama-1.1B

TinyLlama is a compact open-source language model with distinctive characteristics:

- Approximately 1.1 billion parameters
- Trained on a large corpus of text and code
- Designed for efficiency and performance on resource-constrained devices
- Aims to provide competitive performance with minimal computational requirements

Example output for TinyLlama:

```
### Instruction:
Thursday, no rain and warm, route 4206, scheduled at 00:20 and started at 00:17.
The occupancy level at bust stop 678 is:

### Response:
I have no information about the route 4206 and the scheduled time.
However, I can provide you with the information about the bust stop 678.
The occupancy level at bust stop 678 is 100%.
```

### Experimental Observations

#### Response Patterns

Each model exhibited distinct response behaviors:

1. **Mistral-7b**:

   - Consistently generated JSON-formatted responses
   - Frequently included comprehensive metadata
   - Demonstrated most structured and predictable output format
   - Showed ability to adapt to complex instruction sets

2. **Phi-2**:

   - Similar to Mistral-7b, produced JSON-structured responses
   - Included explanatory sections detailing the response structure
   - Less consistent in metadata inclusion compared to Mistral-7b

3. **TinyLlama-1.1B**:
   - Generated more verbose responses
   - Often included debug information
   - Tendency to provide narrative explanations alongside the occupancy level

### Challenges in Fine-Tuning

The experimental results highlighted several key challenges:

1. **Inconsistent Response Formats**: Models frequently deviated from the simple numeric (0, 1, 2) occupancy level prediction
2. **Metadata Overload**: Tendency to include extraneous information instead of focusing on the core prediction task
3. **Interpretation Ambiguity**: Responses like "100%" instead of precise occupancy levels

### Technical Implementation

We maintained our original technical stack:

- **Framework**: PyTorch
- **Fine-tuning Optimization**: Unsloth
- **Model Management**: HuggingFace
- **Hardware**: Intel i7 with 32GB RAM, NVIDIA RTX 3060 GPU

### Project Repository

The complete source code and experimental setup for this research can be found in the project's GitHub repository:

- **Repository**: [Bus Occupancy Prediction](https://github.com/btoffoli/bus_occupancy_prediction)
- **Access**: Open-source implementation of the fine-tuning approach
- **Contents**: Includes data preprocessing scripts, model configurations, and experimental notebooks

### Recommendations for Future Work

1. **Response Constraints**: Implement stricter output formatting guidelines
2. **Fine-tuning Techniques**:
   - Develop more precise instruction templates
   - Augment training data with clear, consistent response formats
3. **Model-Specific Optimization**:
   - Create model-specific post-processing to standardize outputs
   - Explore prompt engineering techniques to guide more accurate predictions
4. **Leverage Mistral-7b Strengths**:
   - Utilize its JSON generation capabilities
   - Develop custom parsing methods to extract precise occupancy levels
   - Explore fine-tuning approaches that capitalize on its structured output tendencies

### Conclusion

While the results demonstrate the potential of using large language models for transportation occupancy prediction, significant refinement is needed. The experiments reveal that simply fine-tuning these models is insufficient; a more nuanced approach to instruction design, model selection, and output processing is crucial.

The Mistral-7b model showed particular promise with its consistent structured outputs and comprehensive response generation. However, like other models, it requires careful post-processing to extract the precise occupancy prediction.

Future research should focus on developing more robust methodologies that can consistently translate complex transportation data into precise, actionable predictions, with special attention to leveraging the unique strengths of each model, particularly Mistral-7b's structured output capabilities.

### References

Vaswani et al., "Attention is All You Need," NeurIPS 2017.

PyTorch Documentation: https://pytorch.org/docs/stable/index.html

Unsloth Documentation: https://docs.unsloth.ai/

HuggingFace Documentation: https://huggingface.co/docs/transformers/index
