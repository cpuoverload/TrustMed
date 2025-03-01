## Development Guide

1. create a virtual environment

You can use conda to create a virtual environment.

```bash
conda create -n TrustMed python=3.10 -y
conda activate TrustMed
```

2. install dependencies

```bash
poetry install
```

3. start the project

backend:

```bash
PYTHONPATH=src python src/api/main.py
```

frontend:

```bash
streamlit run src/webui/streamlit_app.py
```

evaluation:

```bash
PYTHONPATH=src python src/evaluation/create_evaluation_dataset.py
PYTHONPATH=src python src/evaluation/evaluate_rag.py
```
