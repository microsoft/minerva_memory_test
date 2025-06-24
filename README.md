# Introduction 
This repository contains the code for the paper "Minerva: A Programmable Memory Test Benchmark for Language Models" [(PDF)]().

Minerva is a programmable benchmark designed for evaluating how effectively Large Language Models (LLMs) utilize their memory/context. The benchmark provides a structured way to assess various memory-related capabilities of LLMs.

## Test Categories
Minerva comprises six categories of memory tests:

- Search
- Recall and Edit
- Match and Compare
- Spot the Differences
- Compute on Sets and Lists
- Stateful Processing

Plus composite tests that integrate multiple atomic skills to simulate real word scenarios:

- Processing Data Blocks
- Theory of Mind

In total, Minerva consists of 21 distinct tasks spanning these categories.

## Benchmark Snapshot

A complete snapshot of the benchmark dataset used in the paper is available in the `resource/minerva_snapshot` directory.

## Programmability

Minerva is a fully programmable benchmark that allows researchers to customize and extend the test suite. Users can leverage the provided code to generate new test samples with varying parameters, enabling more thorough and tailored evaluations of LLM memory capabilities.

# Quick Start

## Generate Tests

To generate new memory test data:


```python
# Generate all tests
python src/generate_test.py --output_dir ./memory_tests

# Generate specific category tests
python src/generate_test.py --output_dir ./memory_tests --task_category recall_and_edit

# Generate a specific test
python src/generate_test.py --output_dir ./memory_tests --task_name snapshot_unique_words

# List all available tasks
python src/generate_test.py --list-tasks
```

## Run Evaluation

To evaluate an LLM on the memory tests:

We provide a sample script for calling LLM API with Azure OpenAI API.

Please first set up your Azure credentials in `src/azure_api_config.yaml`.


```python
# Run all tests with specific model
python src/run_test.py --task_dir ./memory_tests --result_dir ./results --model_name gpt-4o --llm_aip_config src/azure_api_config.yaml

# Run specific category
python src/run_test.py --task_dir ./memory_tests --result_dir ./results --task_category search

# Run specific test
python src/run_test.py --task_dir ./memory_tests --result_dir ./results --task_name string_search_word
```

# Citation

If you use Minerva in your research, please cite:

```
@misc{xia2025minervaprogrammablememorytest,
      title={Minerva: A Programmable Memory Test Benchmark for Language Models}, 
      author={Menglin Xia and Victor Ruehle and Saravan Rajmohan and Reza Shokri},
      year={2025},
      eprint={2502.03358},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.03358}, 
}
```

# License
This project is licensed under the MIT License - see the LICENSE file for details.