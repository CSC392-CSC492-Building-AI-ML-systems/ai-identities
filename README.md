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
├── response_classifier
│   ├── compare_all_models.py
│   ├── compare.py
│   ├── constants.py
│   ├── data.py
│   ├── get_model_data.bash
│   ├── get_model_data.py
│   ├── model_comparison_results
│   ├── my_results_saved
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
- results:
  - A directory containing JSON files for each pair of model and temperature
  - The JSON files contain the LLM response distribution for each prompt as JSON objects
  - Filenames are structured as model-name_results_<temp>.json where <temp> is one of 0.0, 0.5, 1.0

# xwiki/ directory
