import json
from pathlib import Path
import requests

import numpy as np
import pandas as pd

import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import AntPath, Fullscreen, MarkerCluster

from src.tsp import tspModel


st.set_page_config(
    layout="wide", page_title="TSP Solver", initial_sidebar_state="expanded"
)

filepath = (
    Path.cwd() / "traveling-salesman-problem" / "data" / "available_countries.json"
)
with open(filepath, "r") as fp:
    LIST_OF_COUNTRIES = json.load(fp)


DEFAULTS = {
    "solutionFound": False,
    "obj_val": 0,
    "sol": None,
    "path": [],
    "stats": {},
    "list_of_cities": ["None"],
    "city_df": pd.DataFrame(columns=["row.city", "row.latitude", "row.longitude"]),
    "random_seed": 42,
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


def construct_path(start_city):
    if st.session_state["solutionFound"]:
        sol = st.session_state["sol"]
        first_connection = sol[(sol.n1 == start_city) | (sol.n2 == start_city)].iloc[0]
        path = [
            start_city,
            first_connection.n1
            if first_connection.n2 == start_city
            else first_connection.n2,
        ]

        while path[-1] != path[0]:
            current_node = path[-1]
            previous_node = path[-2]

            connected_rows = sol[(sol.n1 == current_node) | (sol.n2 == current_node)]

            for _, row in connected_rows.iterrows():
                next_node_candidate = row.n1 if row.n2 == current_node else row.n2

                if next_node_candidate != previous_node:
                    path.append(next_node_candidate)
                    break

        return path

    return []


def randomize_cities():
    new_seed = np.random.randint(1, 100)
    st.session_state["random_seed"] = new_seed
    reset_city_data()
    return new_seed


def reset_city_data():
    """Resets city data and the downstream TSP solution."""
    for key in [
        "city_df",
        "list_of_cities",
        "solutionFound",
        "obj_val",
        "path",
        "stats",
    ]:
        st.session_state[key] = DEFAULTS[key]
    st.info("Inputs changed. City list has been updated.")


@st.cache_data(show_spinner=False)
def fetchCities(country):
    base_url = "https://datasets-server.huggingface.co/filter"
    params = {
        "dataset": "jamescalam/world-cities-geo",
        "config": "default",
        "split": "train",
        "where": f"country='{country}'",
        "length": 100,
    }

    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        st.error("Something went wrong while fetching the cities.")

    url = f"https://services9.arcgis.com/l9yXFvhjz46ekkZV/arcgis/rest/services/Countries_Centroids/FeatureServer/0/query?where=COUNTRY+%3D+%27{country}%27&outFields=*&f=pgeojson"

    country_center = requests.get(url)
    if country_center.status_code != 200:
        st.error(
            "Something went wrong while fetching coordinates for the selected country."
        )

    country_centroid = country_center.json()["features"][0]["properties"]

    return response.json()["rows"], [
        country_centroid["latitude"],
        country_centroid["longitude"],
    ]


def solve_tsp(city_df, distance_df, **config):
    obj_val = 0
    stats = pd.DataFrame(columns=["Value"])
    start_city = config["start_city"]

    try:
        sol_list, model = tspModel(city_df, distance_df, maxnodes=config["maxnodes"])
        st.session_state["sol"], time_taken = sol_list
    except Exception as e:
        if e.return_code == 7:
            st.error(
                "Solver returned with status: LicenseError. This might mean you have exceeded the Demo License limit."
            )
            return
        else:
            raise Exception(e)

    if model.solve_status.value not in [1, 2, 3, 5, 8]:
        st.error(
            f"Solution not found. Solver returned with status: {model.solve_status}"
        )
        st.session_state["solutionFound"] = False
    else:
        st.session_state["solutionFound"] = True
        obj_val = round(model.objective_value * 100, 2)  # scale km
        stats_data = {
            "statistics": {
                "Objective Value": obj_val,
                "Solve Time (s)": round(time_taken, 4),
                "Status": model.status.name,
                "#Variables": int(model.num_variables),
                "#Constraints": int(model.num_equations),
            }
        }
        stats = pd.DataFrame.from_dict(
            stats_data["statistics"], orient="index", columns=["Value"]
        )
        stats["Value"] = stats["Value"].astype(str)

    st.session_state["obj_val"] = obj_val
    st.session_state["path"] = construct_path(start_city)
    st.session_state["stats"] = stats


def _make_duplicates_unique(df: pd.DataFrame, mask):
    """
    Helper function to remove duplicates from city names.
    Some cuntries can have same name for a city but they are distinct cities.
    """
    dup_indices: list = df.index[mask]
    for i, idx in enumerate(dup_indices, start=1):
        original = df.at[idx, "row.city"]
        df.at[idx, "row.city"] = f"{original}{i}"

    return df


def prepInput():
    country_centroid = [51.1657, 10.4515]  # DE

    with st.sidebar:
        selected_country = st.selectbox(
            "Choose a country",
            LIST_OF_COUNTRIES,
            index=None,
            placeholder="Type or select a country..",
            help="This tries to fetch 100 cities with their geolocation data from a Kaggle dataset",
            on_change=reset_city_data,
        )

        number_of_cities = st.number_input(
            "Number of cities to consider for TSP",
            min_value=1,
            max_value=63,
            value=10,
            on_change=reset_city_data,
            help="More than 63 cities exceed the demo license limitation."
        )

        if selected_country:
            with st.spinner("Fetching Cities..."):
                city_data, country_centroid = fetchCities(selected_country)
                city_df = pd.json_normalize(city_data)
                city_df = city_df[["row.city", "row.latitude", "row.longitude"]]
                try:
                    city_df = city_df.sample(
                        number_of_cities, random_state=st.session_state["random_seed"]
                    ).reset_index(drop=True)
                    mask = city_df.duplicated(subset=["row.city"], keep="first")
                    if mask.any():
                        city_df = _make_duplicates_unique(city_df, mask)
                    st.session_state["list_of_cities"] = city_df["row.city"].to_list()
                    st.session_state["city_df"] = city_df
                except ValueError:
                    raise Exception(
                        "Maximum nodes is greater than number of cities available. Reduce maximum number of nodes."
                    )

        startCity = st.selectbox(
            label="Start City",
            options=[]
            if "None" in st.session_state["list_of_cities"]
            else st.session_state["list_of_cities"],
            index=None if "None" in st.session_state["list_of_cities"] else 0,
        )

        st.write("")
        st.button(
            "Fetch New Cities",
            type="secondary",
            on_click=randomize_cities,
        )

    input_config = {
        "maxnodes": number_of_cities,
        "start_city": startCity,
        "centroid": country_centroid,
    }

    return input_config


@st.cache_data(show_spinner=False)
def euclidean_distance_matrix(coords):
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    return dist_matrix


@st.cache_data
def plot_solution(
    city_df: pd.DataFrame, centroid: list[float], start_city: str, path: list
):
    country_map = folium.Map(location=centroid, zoom_start=5)

    if len(city_df):
        marker_cluster = MarkerCluster().add_to(country_map)
        for name, lat, long in city_df.itertuples(index=False, name=None):
            if name == start_city:
                folium.Marker(
                    [lat, long],
                    popup=name,
                    icon=folium.Icon(color="green", prefix="fa", icon="location-dot"),
                ).add_to(country_map)
            else:
                # Standard marker for others
                folium.Marker(
                    [lat, long], popup=name, icon=folium.Icon(icon="info-sign")
                ).add_to(marker_cluster)

    if len(path):
        path_df = pd.DataFrame({"row.city": path})
        final_path = path_df.merge(city_df, on="row.city", how="left")

        AntPath(
            locations=final_path[["row.latitude", "row.longitude"]],
            color="#0099FF",
            delay=1000,
            dash_array=[20, 30],
            weight=5,
            opacity=1,
        ).add_to(country_map)

    Fullscreen(
        position="topright",
        title="Fullscreen",
        title_cancel="Exit",
        force_separate_button=True,
    ).add_to(country_map)

    return country_map


def main():
    st.title("GAMSPy TSP Solver")

    config = prepInput()
    city_df: pd.DataFrame = st.session_state["city_df"]
    start_city = config["start_city"]

    with st.sidebar:
        st.divider()
        btn1, btn2 = st.columns(2)

        with btn1:
            run_opt = st.button("Run Optimization", type="primary")

        with btn2:
            st.button("Reset solver", type="secondary", on_click=reset_city_data)

    country_centroid = config["centroid"]

    col1, col2 = st.columns(2)

    with col2:
        output_placeholder = st.empty()
        if run_opt:
            dist_mat = euclidean_distance_matrix(
                city_df[["row.latitude", "row.longitude"]].to_numpy()
            )
            dist_df = pd.DataFrame(
                dist_mat, index=city_df["row.city"], columns=city_df["row.city"]
            )
            distance_df = dist_df.reset_index().melt(
                id_vars="row.city", var_name="to_city", value_name="distance"
            )
            with st.spinner("Solving TSP with DFJ formulation..."):
                output_placeholder.empty()
                if city_df.empty:
                    raise Exception("Select a country first")
                solve_tsp(city_df, distance_df, **config)

        if st.session_state["solutionFound"]:
            with output_placeholder.container():
                st.success("Solution Found!")
                st.markdown("### Optimal Tour:")
                st.write(" ➡️ ".join(construct_path(start_city)))
                st.markdown("### Total distance travelled:")
                st.markdown(
                    f"{st.session_state['obj_val']} km",
                )
                st.markdown("### Model Statistics:")
                st.dataframe(st.session_state["stats"])
        else:
            output_placeholder.empty()

    with col1:
        path = st.session_state.get("path", [])
        country_map = plot_solution(city_df, country_centroid, start_city, path=path)
        with st.spinner("Plotting solution path..."):
            st_folium(country_map, use_container_width=True, height=700)


if __name__ == "__main__":
    main()
