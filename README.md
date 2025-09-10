# LLM Identifier & Wiki

A web application to identify unknown LLMs by comparing their responses to a library of known 
LLMs. The web application also contains a built-in wiki with curated LLM info. This repository 
contains production services, data collection tooling, experimental scripts, and a frontend
for interacting with the classifier and wiki.

**Deployed website link:** https://llmdetective.ca/

**White paper:** [ai-identities_summer2025_white_paper.pdf](ai-identities_summer2025_white_paper.pdf)

Check out the demo video below for a walkthrough of our application's main features! 

https://github.com/user-attachments/assets/320cb6bd-cb3b-4d38-a319-7ee638069f32

## Table of contents

1. [Repository structure](#repository-structure)
2. [Quick start](#quick-start)
3. [Where to find more detail](#where-to-find-more-detail)


# Repository structure

```
├── classifier_service/       # Production classification backend
├── frontend/                 # Web application frontend files
├── response_classifier/      # Classifier methods: data, scripts, results
├── xwiki/                    
├── ai-identities_summer2025_white_paper.pdf  # White paper detailing the project and final web application              
└── README.md                 # (this file)
```

| Folder         | Description                                                                                                                               |
| -------------- |-------------------------------------------------------------------------------------------------------------------------------------------|
| `classifier_service/` | Production-ready classification service code. This is the core backend service that takes in responses and returns predicted LLM matches. |
| `frontend/`    | Frontend files for the deployed web application (includes home page, LLM fingerprinting tool, and the LLM wiki)
| `response_classifier/` | Data collection, analysis scripts, model experiments, configs, and results for classifier methods research.                               |                             

Each of the folders above should contain its own `README.md` with more detailed explanation and instructions.

# Quick start

**How to clone the repository**

```bash
git clone git@github.com:CSC392-CSC492-Building-AI-ML-systems/ai-identities.git
cd ai-identities
```

# Where to find more detail

Each major folder should contain a `README.md` with detailed explanations of its content. Start with:

* `classifier_service/README.md` — classifier service used by the web application
* `frontend/README.md` — web application frontend
* `response_classifier/README.md` — dataset, data collection scripts, classifier method experiment files, and config files
