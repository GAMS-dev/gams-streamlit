# GAMS-Streamlit

This repository contains streamlit applications for various optimization problems.

## Modeling

The optimization problems are modeled and solved using either GAMSPy or GAMS control API. The model for each problem is available in their respective `src` directory.

## Usage

The [requirements.txt](https://github.com/GAMS-dev/gams-streamlit/blob/gams-live/requirements.txt) in the `gams-live` branch contains packages specific to applications that use **GAMS control API** and similarly the [requirements.txt](https://github.com/GAMS-dev/gams-streamlit/blob/gamspy-live/requirements.txt) in the `gamspy-live` branch contains packages specific to applications using **GAMSPy**. This is important because streamlit requires the `requirements.txt` file to be present in the root of the project.

To install the packages run,

```bash
pip install -r requirements.txt
```

You can then run the streamlit application locally by,

```bash
streamlit run <app-name>.py
```

The application should be available at <http://localhost:8501>

You can also run the standalone model from the `src` directory by,

```bash
python <src-name>.py
```
## Demo License

The applications run on a Demo License which has a limit of 2000 variables and constraints.

Following is the list of examples that are available in the streamlit community.

## Streamlit applications

| # | Problem            | Streamlit App                                                                      | Model                                                            | Backend          |
| - | ------------------ | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ---------------- |
| 1 | Cutting stock      | [https://gams-cutstock.streamlit.app/](https://gams-cutstock.streamlit.app/)       | [cutstock.py](./cutstock-problem/src/cutstock.py)         | GAMS control API |
| 2 | Traveling Salesman | [https://gamspy-tsp.streamlit.app/](https://gamspy-tsp.streamlit.app/)             | [tsp.py](./traveling-salesman-problem/src/tsp.py)                 | GAMSPy           |
| 3 | Transportation     | [https://gamspy-transport.streamlit.app/](https://gamspy-transport.streamlit.app/) | [transport.py](./transportation-problem/src/transport.py) | GAMSPy           |
