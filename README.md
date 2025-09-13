# LLM Table Evaluation

This repository contains code for evaluating how Large Language Models handle tabular data under different formatting conditions.

## Overview

The project tests how LLMs perform when extracting information from tabular data with various formatting styles. It measures accuracy across different sample sizes and formatting settings.

## Why Useful?

Through evaluating Data Representations, Data Quality, and Prompt characteristics, we aim to establish a systematic, generalizable way to evaluate LLMs by building evaluation pipelines to help select and fine-tune models for specific use cases based on their different characteristics their data and prompt has.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/llm-table-eval.git
   cd llm-table-eval
   ```

2. Install dependencies:
   ```
   pip install
   ```

3. Add your OpenAI API key in `experiment_general_settings.py`:
   ```python
   os.environ["OPENAI_API_KEY"] = "your-api-key-here"
   ```

## Dataset

The project uses the `california_schools.csv` dataset, which contains information about schools in California including:
- School names
- County codes (cdscode)
- County names
- Other school-related information

## Method

Given a dataset of sample sizes [100, 500, 1000, 2000, 3000, 5000], we select the middle row as the target row that contains information about a school. We then have a query to select the value of the inquired column based on the school name column and county name column. Then the task is to extract the information about the school name, county name, and inquired column from the sampled dataset in the particular format. We compare this result and run it 100 times to reduce variation in accuracy.

With this setting, we manipulate selected independent variables (Data Representations, Data Quality, and Prompt Characteristics) and observe the results. By plotting the results based on sample size and accuracy, we can observe how the LLM reacts to the given task and identify trends as we adjust independent variables.

## Running Experiments

1. Configure experiment settings in `experiment_general_settings.py`:
   - Modify `inquired_column`, `school_column`, and `county_column` as needed
   - Set `repeat_times` for the number of experiment repetitions
   - Configure `sample_sizes` to test different dataset sizes
   - Uncomment desired settings in the `settings` dictionary

2. Run the experiment:
   ```
   python experiment_general_settings.py
   ```

3. Results will be saved in a timestamped folder with:
   - CSV file with detailed results
   - PNG graph visualizing accuracy across sample sizes
   - Summary text file

## Experiment Settings

The code supports various data formatting settings to test how LLMs handle different presentations of tabular data:

- Baseline: Standard presentation with random symbols
- Sorted School: Data sorted by school name
- Group by County: Schools grouped by county
- Group by County with Headers: County headers with highlighting
- Highlight Same County: Highlighting rows with the same county
- Group Sort County Sort School: Grouped by county with sorting
- JSON Format: Data in JSON-like format
- Column Distance: Testing impact of column positioning
- Remove Random Symbols: Clean data without noise
- Predicate Count: Adding additional predicates about target rows

## Output

The experiment produces:
- Accuracy metrics for each setting and sample size
- Error analysis (API errors, prediction errors, etc.)
- Visualizations comparing performance across settings
- Detailed logs of model responses

## Information

**Dataset Source:** https://bird-bench.github.io/

**Acknowledgments:** Special thanks to Sepanta Zeighami, Aditya Parameswaran, and Shreya Shankar at UC Berkeley EPIC Lab for their guidance and support throughout this project!