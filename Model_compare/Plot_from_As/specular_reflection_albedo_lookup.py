from pathlib import Path
import sys

import numpy as np

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

try:
    from .specular_reflection import specular_reflection_phase_curve
except ImportError:
    from specular_reflection import specular_reflection_phase_curve

try:
    from .parameters import PPs
except ImportError:
    from parameters import PPs


DEFAULT_AN_VALUES = np.linspace(0.0, 0.999, 10000)
DEFAULT_THETA_GRID = np.linspace(0.0, np.pi, 4001)
DEFAULT_TABLE_PATH = Path(__file__).with_name("specular_reflection_albedo_lookup.csv")


def specular_ag_as_from_an(
    a_normal,
    theta_grid=DEFAULT_THETA_GRID,
    rp2rs=PPs.Rp2Rs,
    alpha=PPs.alpha,
    inc=90.0,
    a_over_rp=None,
):
    if a_over_rp is None:
        a_over_rp = PPs.semi_axis / PPs.Rp

    theta_grid = np.asarray(theta_grid, dtype=float)
    if theta_grid[0] < 0.0 or theta_grid[-1] < np.pi:
        raise ValueError("theta_grid must cover the interval [0, pi].")

    if a_normal <= 0.0:
        return {
            "An": float(a_normal),
            "Ag": 0.0,
            "As": 0.0,
            "q": np.nan,
        }

    flux_ppm = specular_reflection_phase_curve(
        theta_grid,
        a_normal,
        rp2rs=rp2rs,
        inc=inc,
        alpha=alpha,
        normalize=False,
    )
    flux_ratio = flux_ppm * 1e-6
    ag = flux_ratio[-1] * a_over_rp**2
    a_s = 2.0 * a_over_rp**2 * np.trapz(flux_ratio * np.sin(theta_grid), theta_grid)
    q = a_s / ag if ag > 0.0 else np.nan

    return {
        "An": float(a_normal),
        "Ag": float(ag),
        "As": float(a_s),
        "q": float(q),
    }


def build_specular_albedo_lookup_table(
    an_values=DEFAULT_AN_VALUES,
    theta_grid=DEFAULT_THETA_GRID,
    rp2rs=PPs.Rp2Rs,
    alpha=PPs.alpha,
    inc=90.0,
    a_over_rp=None,
):
    an_values = np.asarray(an_values, dtype=float)
    table = np.empty((an_values.size, 4), dtype=float)

    for index, a_normal in enumerate(an_values):
        row = specular_ag_as_from_an(
            a_normal,
            theta_grid=theta_grid,
            rp2rs=rp2rs,
            alpha=alpha,
            inc=inc,
            a_over_rp=a_over_rp,
        )
        table[index, 0] = row["An"]
        table[index, 1] = row["Ag"]
        table[index, 2] = row["As"]
        table[index, 3] = row["q"]

    return table


def save_specular_albedo_lookup_table(
    output_path=DEFAULT_TABLE_PATH,
    an_values=DEFAULT_AN_VALUES,
    theta_grid=DEFAULT_THETA_GRID,
    rp2rs=PPs.Rp2Rs,
    alpha=PPs.alpha,
    inc=90.0,
    a_over_rp=None,
):
    output_path = Path(output_path)
    table = build_specular_albedo_lookup_table(
        an_values=an_values,
        theta_grid=theta_grid,
        rp2rs=rp2rs,
        alpha=alpha,
        inc=inc,
        a_over_rp=a_over_rp,
    )
    np.savetxt(
        output_path,
        table,
        delimiter=",",
        header="An,Ag,As,q",
        comments="",
        fmt="%.12e",
    )
    return output_path


def load_specular_albedo_lookup_table(table_path=DEFAULT_TABLE_PATH):
    return np.genfromtxt(table_path, delimiter=",", names=True)


def _interpolate_an_from_column(target, target_column, table):
    x = np.asarray(table[target_column], dtype=float)
    y = np.asarray(table["An"], dtype=float)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    target = np.asarray(target, dtype=float)
    if np.any(target < x_sorted[0]) or np.any(target > x_sorted[-1]):
        raise ValueError(
            f"Target values must lie within [{x_sorted[0]}, {x_sorted[-1]}]."
        )
    return np.interp(target, x_sorted, y_sorted)


def interpolate_specular_an_from_ag(ag, table=None, table_path=DEFAULT_TABLE_PATH):
    if table is None:
        table = load_specular_albedo_lookup_table(table_path)
    return _interpolate_an_from_column(ag, "Ag", table)


def interpolate_specular_an_from_as(a_s, table=None, table_path=DEFAULT_TABLE_PATH):
    if table is None:
        table = load_specular_albedo_lookup_table(table_path)
    return _interpolate_an_from_column(a_s, "As", table)


def main():
    output_path = save_specular_albedo_lookup_table()
    table = load_specular_albedo_lookup_table(output_path)
    print(f"Saved lookup table to: {output_path}")
    print(f"Rows: {table.size}")
    print(f"An range: {table['An'][0]} -> {table['An'][-1]}")
    print(f"Ag range: {table['Ag'][0]} -> {table['Ag'][-1]}")
    print(f"As range: {table['As'][0]} -> {table['As'][-1]}")


if __name__ == "__main__":
    main()
