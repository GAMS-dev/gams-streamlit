from __future__ import annotations

import pandas as pd
from gamspy import (
    Container,
    Equation,
    Model,
    Parameter,
    Sense,
    Set,
    Sum,
    Variable,
)


def transport_model(
    capacities: list[tuple[str, int]],
    demands: list[tuple[str, int]],
    distance: pd.DataFrame,
    freight_cost: float = 90,
) -> tuple[pd.DataFrame, Model]:
    m = Container()

    # Set
    i = Set(
        m,
        name="i",
        description="canning plants",
    )
    j = Set(
        m,
        name="j",
        description="markets",
    )

    # Data
    a = Parameter(
        m,
        name="a",
        domain=i,
        records=capacities,
        domain_forwarding=True,
        description="capacity of plant i in cases",
    )
    b = Parameter(
        m,
        name="b",
        domain=j,
        records=demands,
        domain_forwarding=True,
        description="demand at market j in cases",
    )
    d = Parameter(
        m,
        name="d",
        domain=[i, j],
        records=distance,
        description="distance in thousands of miles",
    )
    c = Parameter(
        m,
        name="c",
        domain=[i, j],
        description="transport cost in thousands of dollars per case",
    )
    c[i, j] = freight_cost * d[i, j] / 1000

    # Variable
    x = Variable(
        m,
        name="x",
        domain=[i, j],
        type="Positive",
        description="shipment quantities in cases",
    )

    # Equation
    supply = Equation(
        m,
        name="supply",
        domain=i,
        description="observe supply limit at plant i",
    )
    demand = Equation(
        m, name="demand", domain=j, description="satisfy demand at market j"
    )

    supply[i] = Sum(j, x[i, j]) <= a[i]
    demand[j] = Sum(i, x[i, j]) >= b[j]

    transport = Model(
        m,
        name="transport",
        equations=m.getEquations(),
        problem="LP",
        sense=Sense.MIN,
        objective=Sum((i, j), c[i, j] * x[i, j]),
    )
    transport.solve()

    print(transport.objective_value)
    print(transport.status)

    df = x.l.records
    return df[df["level"] > 0.5], transport


if __name__ == "__main__":
    capacities = [("seattle", 350), ("san-diego", 600)]
    demands = [("new-york", 325), ("chicago", 300), ("topeka", 275)]

    distances = pd.DataFrame(
        [
            ["seattle", "new-york", 2.5],
            ["seattle", "chicago", 1.7],
            ["seattle", "topeka", 1.8],
            ["san-diego", "new-york", 2.5],
            ["san-diego", "chicago", 1.8],
            ["san-diego", "topeka", 1.4],
        ],
        columns=["from", "to", "distance"],
    ).set_index(["from", "to"])

    solution, model = transport_model(capacities, demands, distances.reset_index(), freight_cost=90)
    print(solution)
