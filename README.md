# LLM Table Evaluation

This repository contains code for evaluating how Large Language Models handle tabular data under different formatting conditions.

## Overview

The project tests how LLMs perform when extracting information from tabular data with various formatting styles. It measures accuracy across different sample sizes and formatting settings.

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

**Acknowledgments:** Special thanks to my mentors Sepanta Zeighami, Aditya Parameswaran, and Shreya Shankar for their guidance and support throughout this project!