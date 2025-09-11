# GAMS-Streamlit

This repository contains streamlit applications for various optimization problems. 

## Modeling

The optimization problems are modeled and solved using either GAMSPy or GAMS control API. The source code for each problem is available in the [src](./src/) directory.

## Usage

The [requirements.txt](./requirements.txt) file contains all the necessary package required to run the streamlit application locally or test the models directly.

To install the packages run,

```bash
pip install -r requirements.txt
```

You can then run the streamlit application locally by,

```bash
streamlit run <app-name>.py
```
The application should be available at http://localhost:8501

Following is the list of examples that are available in the streamlit community.

## Streamlit applications

1) Cutting stock problem
2) Traveling Salesman problem
