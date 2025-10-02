import json
from pathlib import Path

import numpy as np
import pandas as pd

import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import AntPath, Fullscreen, MarkerCluster

from src.transport import transport_model


st.set_page_config(
    layout="wide", page_title="Transportation Problem", initial_sidebar_state="expanded", page_icon="🚚"
)

if "all_city_data" not in st.session_state:
    filepath = (
        Path.cwd() / "transportation-problem" / "data" / "us_cities_100.json"
    )  # for streamlit
    # filepath = Path.cwd() / "data" / "us_cities_100.json",  # local runs or run from root directory
    with open(filepath, "r") as fp:
        city_data = json.load(fp)
        st.session_state["all_city_data"] = pd.DataFrame.from_dict(
            city_data, orient="index", columns=["Latitude", "Longitude"]
        ).reset_index()

DEFAULTS = {
    "solutionFound": False,
    "obj_val": 0,
    "sol": None,
    "stats": {},
    "suppliers": pd.DataFrame(
        [("Seattle", 350), ("San Diego", 600)], columns=["Name", "Capacity"]
    ),
    "markets": pd.DataFrame(
        [("New York", 325), ("Chicago", 300), ("Topeka", 275)],
        columns=["Name", "Demand"],
    ),
}


def getEntities(suppliers, markets):
    list_of_suppliers_and_markets = (
        suppliers["Name"].values.tolist() + markets["Name"].values.tolist()
    )
    return list_of_suppliers_and_markets


DEFAULTS["filtered_city_df"] = (
    st.session_state["all_city_data"]
    .loc[
        st.session_state["all_city_data"]["index"].isin(
            getEntities(DEFAULTS["suppliers"], DEFAULTS["markets"])
        )
    ]
    .reset_index(drop=True)
)

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


def reset_all():
    """Resets city data and the downstream TSP solution."""
    for key in ["suppliers", "markets", "solutionFound", "obj_val", "stats", "sol"]:
        st.session_state[key] = DEFAULTS[key]
    st.info("Solver Reset.")


def solve_transport(capacities, demands, distance, freight_cost):
    obj_val = 0
    stats = pd.DataFrame(columns=["Value"])

    try:
        solution, model = transport_model(capacities, demands, distance, freight_cost)
        st.session_state["sol"] = solution
    except Exception as e:
        if e.return_code == 7:
            raise Exception(
                "Solver returned with status: LicenseError. This might mean you have exceeded the Demo License limit."
            )
        else:
            raise Exception(e)

    if model.solve_status.value not in [1, 2, 3, 5, 8]:
        st.error(
            f"Solution not found. Solver returned with status: {model.solve_status}"
        )
        st.session_state["solutionFound"] = False
    else:
        st.session_state["solutionFound"] = True
        obj_val = round(model.objective_value, 2)
        stats_data = {
            "statistics": {
                "Objective Value": obj_val,
                "Solve Time (s)": round(model.solve_model_time, 4),
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
    st.session_state["stats"] = stats


def reset_solution():
    """Resets stock data and the downstream TSP solution."""
    for key in [
        "solutionFound",
        "sol",
        "obj_val",
        "stats",
    ]:
        st.session_state[key] = DEFAULTS[key]
    st.toast("Updating input parameter.")


@st.dialog("Add new entities")
def add_entities():
    list_of_all_cities = set(st.session_state["all_city_data"]["index"].values.tolist())
    suppliers = st.session_state["suppliers"]
    markets = st.session_state["markets"]

    list_of_new_suppliers_and_markets = list_of_all_cities - (
        set(getEntities(suppliers, markets))
    )

    isSupplier = st.radio(
        label="Is this city a supplier", options=["Yes", "No"], index=0
    )
    city_name = st.selectbox(
        "Select a city",
        list_of_new_suppliers_and_markets,
        index=None,
        placeholder="Type or select a city...",
    )
    if isSupplier == "Yes":
        param = st.number_input("Enter capacity of the new supplier", value=0, step=1)
        new_row = {"Name": city_name, "Capacity": param}

    elif isSupplier == "No":
        param = st.number_input("Enter demand for the new market", value=0, step=1)
        new_row = {"Name": city_name, "Demand": param}

    if st.button("Submit"):
        if isSupplier == "Yes":
            st.session_state["suppliers"] = pd.concat(
                [st.session_state["suppliers"], pd.DataFrame([new_row])],
                ignore_index=True,
            )
        elif isSupplier == "No":
            st.session_state["markets"] = pd.concat(
                [st.session_state["markets"], pd.DataFrame([new_row])],
                ignore_index=True,
            )
        reset_solution()
        st.rerun()


@st.dialog("Edit entities")
def edit_entities():
    suppliers = st.session_state["suppliers"]
    markets = st.session_state["markets"]

    list_of_suppliers_and_markets = getEntities(suppliers, markets)
    edit_city = st.selectbox(
        label="Select a city to edit",
        options=list_of_suppliers_and_markets,
        index=0,
        placeholder="Type or Select a city to edit...",
    )

    if edit_city in suppliers["Name"].values.tolist():
        new_cap = st.number_input(
            "Enter new capacity for the supplier", value=0, step=1
        )
        if new_cap:
            suppliers.loc[suppliers["Name"] == edit_city, "Capacity"] = new_cap
    elif edit_city in markets["Name"].values.tolist():
        new_demand = st.number_input("Enter new demand for the market", value=0, step=1)
        if new_demand:
            markets.loc[markets["Name"] == edit_city, "Demand"] = new_demand

    if st.button("Update"):
        st.session_state["suppliers"] = suppliers
        st.session_state["markets"] = markets
        reset_solution()
        st.rerun()


@st.dialog("Remove entities")
def remove_entities():
    suppliers = st.session_state["suppliers"]
    markets = st.session_state["markets"]

    list_of_suppliers_and_markets = getEntities(suppliers, markets)
    remove_cities = st.multiselect(
        label="Select one or more cities",
        default=None,
        max_selections=(len(list_of_suppliers_and_markets) - 1),
        options=list_of_suppliers_and_markets,
    )

    if st.button("Remove selected items"):
        remove_supplier = []
        remove_market = []
        for city in remove_cities:
            if city in suppliers["Name"].values.tolist():
                remove_supplier.append(city)
            elif city in markets["Name"].values.tolist():
                remove_market.append(city)
        st.session_state["suppliers"] = suppliers.loc[
            ~suppliers["Name"].isin(remove_supplier)
        ].reset_index(drop=True)
        st.session_state["markets"] = markets.loc[
            ~markets["Name"].isin(remove_market)
        ].reset_index(drop=True)
        reset_solution()
        st.rerun()


def update_city_df():
    st.session_state["filtered_city_df"] = (
        st.session_state["all_city_data"]
        .loc[
            st.session_state["all_city_data"]["index"].isin(
                getEntities(st.session_state["suppliers"], st.session_state["markets"])
            )
        ]
        .reset_index(drop=True)
    )


def prepInput():
    with st.sidebar:
        if st.button("Add new entities", use_container_width=True, icon="➕"):
            add_entities()

        if st.button("Remove entities", use_container_width=True, icon="➖"):
            remove_entities()

        if st.button("Edit entities", use_container_width=True, icon="✂️"):
            edit_entities()

        update_city_df()
        freight_cost = st.number_input(
            label="Freight Cost",
            min_value=0,
            max_value=None,
            value=90,
            step=1,
            on_change=reset_solution,
            help="The cost per unit of shipment between plant `i` and market `j` is derived from `freight_cost * distance(i,j) / 1000`",
        )

    return freight_cost


@st.cache_data(show_spinner=False)
def euclidean_distance_matrix(coords):
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    return dist_matrix


def plot_solution(filtered_city_df: pd.DataFrame, sol):
    suppliers = st.session_state["suppliers"]
    markets = st.session_state["markets"]
    country_map = folium.Map(
        location=[39.8333, -98.5833], zoom_start=5
    )  # center to USA

    marker_cluster = MarkerCluster().add_to(country_map)
    for name, lat, long in filtered_city_df.itertuples(index=False, name=None):
        if name in suppliers["Name"].values.tolist():
            # supplier markers
            folium.Marker(
                [lat, long],
                tooltip=f"{name} (Capacity: {suppliers[suppliers['Name'] == name]['Capacity'].values[0]})",
                icon=folium.Icon(color="blue"),
            ).add_to(country_map)
        elif name in markets["Name"].values.tolist():
            # market markers markers
            folium.Marker(
                [lat, long],
                tooltip=f"{name} (Demand: {markets[markets['Name'] == name]['Demand'].values[0]})",
                icon=folium.Icon(color="green"),
            ).add_to(marker_cluster)

    if sol is not None:
        max_level = max(sol.level)
        for i, j, level in sol.itertuples(index=False):
            selection_i = filtered_city_df[filtered_city_df["index"] == i][
                ["Latitude", "Longitude"]
            ]
            selection_j = filtered_city_df[filtered_city_df["index"] == j][
                ["Latitude", "Longitude"]
            ]
            combined_df = pd.concat([selection_i, selection_j], ignore_index=True)
            weight = level * 8 / max_level

            AntPath(
                locations=combined_df,
                color="#FF2600B0",
                delay=1000,
                dash_array=[20, 30],
                weight=weight,
                opacity=0.8,
                tooltip=f"Level: {level}",
            ).add_to(country_map)

    Fullscreen(
        position="topright",
        title="Fullscreen",
        title_cancel="Exit",
        force_separate_button=True,
    ).add_to(country_map)

    return country_map


def main():
    st.title("GAMSPy Transportation problem solver")

    freight_cost = prepInput()
    suppliers = st.session_state["suppliers"]
    markets = st.session_state["markets"]
    filtered_city_df = st.session_state["filtered_city_df"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### List of Suppliers")
        st.dataframe(suppliers)
        st.write(f"Total supplier capacity: **{suppliers.Capacity.sum()}**")
    with col2:
        st.markdown("### List of Markets")
        st.dataframe(markets)
        st.write(f"Total market demand: **{markets.Demand.sum()}**")

    with st.sidebar:
        st.divider()
        btn1, btn2 = st.columns(2)

        with btn1:
            run_opt = st.button(
                "Run Optimization", type="primary", use_container_width=True
            )

        with btn2:
            st.button(
                "Reset solver",
                type="secondary",
                on_click=reset_all,
                use_container_width=True,
            )

    output_placeholder = st.empty()
    if run_opt:
        dist_mat = euclidean_distance_matrix(
            filtered_city_df[["Latitude", "Longitude"]].to_numpy()
        )
        dist_df = pd.DataFrame(
            dist_mat, index=filtered_city_df["index"], columns=filtered_city_df["index"]
        )
        distance_df = dist_df.reset_index().melt(
            id_vars="index", var_name="to_city", value_name="Euclidean distance"
        )
        distance_df = distance_df[
            (distance_df["index"].isin(suppliers["Name"]))
            & (distance_df["to_city"].isin(markets["Name"]))
        ].reset_index(drop=True)

        with st.spinner("Transporting goods, meeting demands 🚚..."):
            output_placeholder.empty()
            if len(filtered_city_df) <= 1:
                raise Exception("Select at least one supplier and a market.")
            solve_transport(
                capacities=list(suppliers.itertuples(index=False, name=None)),
                demands=list(markets.itertuples(index=False, name=None)),
                distance=distance_df,
                freight_cost=freight_cost,
            )

    if st.session_state["solutionFound"]:
        with output_placeholder.container():
            status = st.session_state["stats"].loc["Status"].values[0]
            if status == "OptimalGlobal":
                st.success("Optimal solution found!", icon="✅")
            elif status == "InfeasibleGlobal":
                st.error("Model infeasible!", icon="❌")
            else:
                st.info(f"Model status is {status}", icon="ℹ️")
            st.markdown("### Total Cost:")
            st.markdown(
                f"$ {st.session_state['obj_val']}",
            )
            st.markdown("### Model Statistics:")
            st.dataframe(st.session_state["stats"])
    else:
        output_placeholder.empty()

    country_map = plot_solution(filtered_city_df, sol=st.session_state["sol"])
    with st.spinner("Plotting solution..."):
        st_folium(country_map, use_container_width=True, height=700)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(e)
