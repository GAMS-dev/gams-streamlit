# GAMS-Streamlit

This repository contains streamlit applications for various optimization problems. 

## Modeling

The optimization problems are modeled and solved using either GAMSPy or GAMS control API. The model for each problem is available in their respective `src` directory.

## Usage

The `requirements.txt` file contains all the necessary package required to run the streamlit application locally or test the models directly.

To install the packages run,

```bash
pip install -r requirements.txt
```

You can then run the streamlit application locally by,

```bash
streamlit run <app-name>.py
```
The application should be available at http://localhost:8501


You can also run the standalone model from the `src` directory by,

```bash
python <src-name>.py
```

Following is the list of examples that are available in the streamlit community.

## Streamlit applications


| Problem            | Streamlit App                                                                | Model                                             | Backend          |
| ------------------ | ---------------------------------------------------------------------------- | ------------------------------------------------- | ---------------- |
| Cutting stock      | [https://gams-cutstock.streamlit.app/](https://gams-cutstock.streamlit.app/) | [cutstock.py](./cutstock-problem/src/cutstock.py) | GAMS control API |
| Traveling Salesman | [https://gamspy-tsp.streamlit.app/](https://gamspy-tsp.streamlit.app/)       | [tsp.py](./traveling-salesman-problem/src/tsp.py) | GAMSPy           |
