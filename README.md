# LLM Identifier & Wiki

**A web application to identify unknown LLMs by comparing their responses to a catalog of known LLMs. The web application also contains a built-in wiki with curated LLM info.**

This repository contains production services, data collection tooling, experimental scripts, and a frontend for interacting with the classifier and wiki.

---

## Table of contents

1. [Repository structure](#repository-structure)
2. [Quick start (minimum steps)](#quick-start-minimum-steps)
3. [Development workflow highlights](#development-workflow-highlights)
4. [Where to find more detail](#where-to-find-more-detail)
5. [License](#license)


# Repository structure

```
├── classifier_service/       # Production classification backend
├── frontend/                 # Add short description here
├── response_classifier/      # Classifier methods: data, experiments, plots, scripts, results
├── web-gui-grabber/          # Add short description here
├── xwiki/                    # Add short description here
├── node_modules/             # Add short description here
├── package.json              # Add short description here
└── README.md                 # (this file)
```

| Folder                     | Description                                                                                                                               |
| -------------------------- |-------------------------------------------------------------------------------------------------------------------------------------------|
| **`classifier_service/`**  | Production-ready classification service code. This is the core backend service that takes in responses and returns predicted LLM matches. |
| **`frontend/`**            | Add short description here                                                                                                                |
| **`response_classifier/`** | Data collection, analysis scripts, model experiments, configs, and results for classifier methods research.                               |
| **`web-gui-grabber/`**     | Add short description here                                                                                                                |
| **`xwiki/`**               | Add short description here                                                                                                                |
| **`node_modules/`**        | Add short description here                                                                                                                |

Each of the folders above should contain its own `README.md` with more detailed explanation and instructions.

# Quick start (minimum steps)

1. **How to clone the repository**

```bash
git clone git@github.com:CSC392-CSC492-Building-AI-ML-systems/ai-identities.git
cd ai-identities
```

# Development workflow highlights

* **Collecting data**
  * Use `web-gui-grabber` to collect responses from target LLM UIs. Store raw outputs under `web-gui-grabber/browsing-data/` and sync to `response_classifier/data/` if needed.

* **Experimentation & training**
  * All classifier method experiment configs, datasets, analysis, and results lives in `response_classifier/`. Use the provided config files and scripts to reproduce experiments.

* **Production classifier**
  * `classifier_service/` is where the production inference logic lives. Keep model-loading, scoring, and API stable here — experiments should be prototyped in `response_classifier/` and then ported (or reimplemented) into `classifier_service/`.

# Where to find more detail

Each major folder should contain a `README.md` with detailed explanations of its content. Start with:

* `classifier_service/README.md` — 
* `frontend/README.md` — 
* `response_classifier/README.md` — dataset, data collection scripts, classifier method experiment files, and config files
* `web-gui-grabber/README.md` — ...
* `xwiki/README.md` — how the wiki works and how we can populate it


# License

* **License:** ...
