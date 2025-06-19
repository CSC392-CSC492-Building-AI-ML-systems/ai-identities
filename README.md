# Repository Structure
```text
├── frontend
│   ├── app
│   ├── components
│   ├── eslint.config.mjs
│   ├── next.config.ts
│   ├── package.json
│   ├── package-lock.json
│   ├── postcss.config.js
│   ├── public
│   ├── README.md
│   ├── styles
│   ├── tailwind.config.js
│   └── tsconfig.json
├── README.md
├── response_classifier
│   ├── compare_all_models.py
│   ├── compare.py
│   ├── constants.py
│   ├── data.py
│   ├── get_data_by_temp.bash
│   ├── get_model_data.bash
│   ├── get_model_data.py
│   ├── model_comparison_results
│   ├── models.txt
│   └── results
└── xwiki
    └── docker-compose.yml
```
# frontend/ directory

# response_classifier/ directory
- compare.py:
  - Takes two arguments; a target json file and the path to the results directory which contains all the json files of LLM responses
  - Compares the LLM response distribution of the target json file against all other json files in the results directory
- compare_all_models.py:
  - Compares each json file in the results directory against all other json files and saves the result/report as a text file in the model_comparison_results/ directory
- constants.py:
  - Contains a single multi-line string where each line is an incomplete sentence that will be used as a prompt for LLM response data collection
- data.py:
  - Computes the number of json objects (which is essentially the number of prompts for which our program was able to collect LLM response data) in each json file in the results/ directory
- get_data_by_temp.bash:
  - Takes LLM model temperature as an argument and executes get_model_data.py to collected LLM responses for the prompts in constants.py
  - Collects LLM response data for all the models listed in models.txt
- get_model_data.bash:
  - Executes get_model_data.py to collected LLM responses for the prompts in constants.py
- get_model_data.py:
  - Processes constants.py to create a set of unique prompts
  - Sends 20 requests to Deepinfra for each prompt
  - Collects LLM responses for all the requests sent
  - Only the first token in the LLM's response is collected
  - Stores the collected data as JSON files in the results/ directory
- model_comparison_results:
  - A directory containing text files for each pair of model and temperature
  - The text files contain the total number of LLM response matches and the matching rate in percentage
  - Filenames are structured as model-name_report_<temp>.txt where <temp> is one of 0.0, 0.5, 1.0
- models.txt:
  - Contains model names to use with get_data_by_temp.bash to collect data for a set of models
- results:
  - A directory containing JSON files for each pair of model and temperature
  - The JSON files contain the LLM response distribution for each prompt as JSON objects
  - Filenames are structured as model-name_results_<temp>.json where <temp> is one of 0.0, 0.5, 1.0

### How to use the bash script to collect LLM response data for many models?
First, note that regardless of which bash script you use to collect LLM response data, your project directory should contain the file response_classifier/keys.txt which includes the API key. 

To easily collect data from many models, you can use the following command and the resulting json files will be saved to the response_classifier/results/ directory. The script will collect data for each model listed in the models.txt file.
- `--temp <temperature>`: (required) LLM temperature to use
- `--testing`: (optional) Limits data collection to 5 unique prompts for a quick check (will be particularly useful to check the collected data of reasoning models)
```bash
./get_data_by_temp.bash [--testing] --temp <temperature>
```

Examples
```bash
# Full prompt set, temperature 0.5
./get_data_by_temp.bash --temp 0.5

# Test run (5 prompts), temperature 0.5
./get_data_by_temp.bash --testing --temp 0.5
```

### How to use the bash script to collect LLM response data for a single model?
Ensure that you have response_classifier/keys.txt which includes the API key.

To collect data for a single model, you can use the following command and the resulting json files will be saved to the response_classifier/results/ directory.
- Currently, users will have to manually set the model name and temperature in the bash script
- `--testing`: (optional) Limits data collection to 5 unique prompts for a quick check (will be particularly useful to check the collected data of reasoning models)
```bash
./get_data_by_temp.bash [--testing]
```

### How to generate the text file reports of LLM responses' comparison?
Simply run the following command and the text file reports will be saved in the response_classifier/model_comparison_results/ directory.

Examples
```bash
python compare_all_models.py
```

# xwiki/ directory
