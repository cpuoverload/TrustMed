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
python main.py
```

frontend:

```bash
streamlit run frontend/streamlit_app.py
```
