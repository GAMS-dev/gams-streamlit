import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.colors

from src.cutstock import cutStockModel

st.set_page_config(
    layout="wide", page_title="CutStock Solver", initial_sidebar_state="expanded"
)


data = {
    "ID": ["Kraft", "Newsprint", "Coated", "Lightweight"],
    "demand": [97, 610, 395, 211],
    "width": [47, 36, 31, 14],
}


DEFAULTS = {
    "solutionFound": False,
    "obj_val": 0,
    "sol": None,
    "stats": {},
    "stock_df": pd.DataFrame(data),
}


for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


def reset_solution():
    """Resets stock data and the downstream TSP solution."""
    for key in [
        "solutionFound",
        "obj_val",
        "stats",
    ]:
        st.session_state[key] = DEFAULTS[key]
    st.info("Solution Reset. Updating list of products.")


@st.dialog("Remove products")
def remove_products():
    stock_df: pd.DataFrame = st.session_state["stock_df"]
    remove_items = st.multiselect(
        label="Select one or more items",
        default=None,
        max_selections=(len(stock_df) - 1),
        options=stock_df["ID"],
    )

    if st.button("Remove selected items"):
        st.session_state["stock_df"] = stock_df.loc[
            ~stock_df["ID"].isin(remove_items)
        ].reset_index(drop=True)
        reset_solution()
        st.rerun()


@st.dialog("Enter Info")
def add_more_products():
    stock_df = st.session_state["stock_df"]
    product_name = st.text_input(
        label="product ID", value=f"Item{len(stock_df) + 1}", max_chars=20
    )
    demand = st.number_input("Enter demand for new product", value=0, step=1)
    width = st.number_input("Enter width of the new product", value=100, step=1)
    if st.button("Submit"):
        new_row = {"ID": product_name, "demand": demand, "width": width}
        st.session_state["stock_df"] = pd.concat(
            [stock_df, pd.DataFrame([new_row])], ignore_index=True
        )
        reset_solution()
        st.rerun()


def plotCutstock(raw_width):
    sol = st.session_state["sol"]
    stock_df = st.session_state["stock_df"]

    width_of_each_product = stock_df.set_index("ID")["width"].to_dict()
    item_names = sorted(
        {item for pattern in sol.values() for item in pattern["itemCut"]}
    )
    color_cycle = plotly.colors.qualitative.Plotly
    item_colors = {
        item: color_cycle[i % len(color_cycle)] for i, item in enumerate(item_names)
    }
    item_colors["Waste"] = "lightgrey"

    patterns = [f"{p} x {int(details['ncuts'])}" for p, details in sol.items()]

    # Prepare data for stacked bars: one trace per item (for legend toggling)
    traces = {
        item: {"x": [], "y": [], "text": [], "color": item_colors[item]}
        for item in item_names + ["Waste"]
    }

    for idx, (_, details) in enumerate(sol.items()):
        y_label = patterns[idx]
        used_length = 0
        # map items for this pattern
        for item in item_names:
            qty = details["itemCut"].get(item, 0)
            item_len = width_of_each_product.get(item, 0)
            total_len = qty * item_len
            traces[item]["x"].append(total_len)
            traces[item]["y"].append(y_label)
            if total_len > 0:
                traces[item]["text"].append(f"{item}: {int(qty)} x {item_len}")
            else:
                traces[item]["text"].append("")
            used_length += total_len
        # Add waste
        waste_len = max(0, raw_width - used_length)
        traces["Waste"]["x"].append(waste_len)
        traces["Waste"]["y"].append(y_label)
        traces["Waste"]["text"].append(f"Waste: {waste_len}" if waste_len > 0 else "")

    fig = go.Figure()

    for item, data in traces.items():
        showlegend = True
        fig.add_trace(
            go.Bar(
                x=data["x"],
                y=data["y"],
                orientation="h",
                marker=dict(color=data["color"]),
                name=item,
                text=data["text"],
                textposition="auto",
                showlegend=showlegend,
            )
        )

    fig.update_layout(
        barmode="stack",
        title="Cutting Pattern Usage & Waste",
        xaxis_title="Stock Length (per pattern)",
        yaxis_title="Cutting Pattern",
        legend_title="Products",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1),
    )

    st.plotly_chart(fig)


def solveCutStock(raw_width, max_pattern):
    stock_df = st.session_state["stock_df"]
    demand = stock_df.set_index("ID")["demand"].to_dict()
    width = stock_df.set_index("ID")["width"].to_dict()

    obj_val = 0
    stats = pd.DataFrame(columns=["Value"])

    try:
        start = time.time()
        [patternFlag, _], obj_val, cut_info = cutStockModel(
            d=demand, w=width, r=raw_width, max_pattern=max_pattern
        )
        tot_time = round(time.time() - start, 2)
        if patternFlag:
            st.info(
                f"Optimal solution not guaranteed. Increase `max_pattern` (currently {max_pattern})."
            )
        st.session_state["sol"] = cut_info
    except Exception as e:
        raise Exception(e)

    st.session_state["solutionFound"] = True
    stats_data = {
        "statistics": {"Objective Value": obj_val, "Solve time (s)": tot_time}
    }
    stats = pd.DataFrame.from_dict(
        stats_data["statistics"], orient="index", columns=["Value"]
    )
    stats["Value"] = stats["Value"].astype(str)

    st.session_state["obj_val"] = obj_val
    st.session_state["stats"] = stats


def prepInput():
    with st.sidebar:
        in1, in2 = st.columns(2)
        with in1:
            raw_width = st.number_input(
                "Raw width",
                min_value=1,
                max_value=1000,
                value=100,
                step=1,
                on_change=reset_solution,
            )

            st.write("")
            if st.button("Add Products"):
                add_more_products()

        with in2:
            max_pattern = st.number_input(
                "Maximum patterns",
                min_value=1,
                max_value=1000,
                value=35,
                step=1,
                on_change=reset_solution,
            )

            st.write("")
            if st.button("Remove Products"):
                remove_products()

    input_config = {
        "rawwidth": int(raw_width),
        "maxpattern": int(max_pattern),
    }

    return input_config


def main():
    st.title("GAMS Cutting Stock Solver")

    config = prepInput()

    with st.sidebar:
        st.divider()
        btn1, btn2 = st.columns(2)

        with btn1:
            run_opt = st.button("Run Optimization", type="primary")

        with btn2:
            st.button("Reset solver", type="secondary", on_click=reset_solution)

    if run_opt:
        with st.spinner("Cutting Stock..."):
            solveCutStock(
                raw_width=config["rawwidth"],
                max_pattern=config["maxpattern"],
            )

    if st.session_state["solutionFound"]:
        plotCutstock(raw_width=config["rawwidth"])
        st.markdown(
            f"### Total number of patterns used: {int(st.session_state['obj_val'])}"
        )
        st.write(st.session_state["stats"])
    else:
        st.markdown("## List of Products")
        st.dataframe(st.session_state["stock_df"])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(e)