from pathlib import Path
import sys
import argparse

import numpy as np

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

try:
    from .specular_reflection_albedo_lookup import (
        DEFAULT_TABLE_PATH,
        interpolate_specular_an_from_ag,
        interpolate_specular_an_from_as,
    )
except ImportError:
    from specular_reflection_albedo_lookup import (
        DEFAULT_TABLE_PATH,
        interpolate_specular_an_from_ag,
        interpolate_specular_an_from_as,
    )


def lambert_an_from_ag(ag):
    return 1.5 * np.asarray(ag, dtype=float)


def lambert_an_from_as(a_s):
    return np.asarray(a_s, dtype=float)


def _match_input_type(result, original_value):
    original_array = np.asarray(original_value)
    result_array = np.asarray(result, dtype=float)
    if original_array.ndim == 0:
        return float(result_array)
    return result_array


def an_from_reflection_albedo(
    value,
    albedo_type="Ag",
    reflection_model="specular",
    table=None,
    table_path=DEFAULT_TABLE_PATH,
):
    """
    Convert a specified Ag or As to the normal-incidence albedo An.

    Parameters
    ----------
    value : float or array-like
        The specified geometric albedo Ag or spherical albedo As.
    albedo_type : {"Ag", "As"}
        Which albedo is being supplied.
    reflection_model : {"specular", "lambert"}
        Reflection model used for the inversion.
    table : structured ndarray, optional
        Preloaded lookup table for specular interpolation.
    table_path : path-like, optional
        CSV path used when loading the specular lookup table.
    """
    albedo_key = str(albedo_type).strip().lower()
    model_key = str(reflection_model).strip().lower()

    if albedo_key not in {"ag", "as"}:
        raise ValueError("albedo_type must be 'Ag' or 'As'.")
    if model_key not in {"specular", "lambert"}:
        raise ValueError("reflection_model must be 'specular' or 'lambert'.")

    if model_key == "lambert":
        if albedo_key == "ag":
            return _match_input_type(lambert_an_from_ag(value), value)
        return _match_input_type(lambert_an_from_as(value), value)

    if albedo_key == "ag":
        result = interpolate_specular_an_from_ag(value, table=table, table_path=table_path)
    else:
        result = interpolate_specular_an_from_as(value, table=table, table_path=table_path)
    return _match_input_type(result, value)


def _build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Convert Ag or As to An for specular or Lambert reflection."
    )
    parser.add_argument(
        "--model",
        choices=["specular", "lambert"],
        required=True,
        help="Reflection model used for the inversion.",
    )
    parser.add_argument(
        "--albedo-type",
        choices=["Ag", "As", "ag", "as"],
        required=True,
        help="Specify whether the input value is Ag or As.",
    )
    parser.add_argument(
        "--value",
        type=float,
        required=True,
        help="Input Ag or As value.",
    )
    parser.add_argument(
        "--table-path",
        default=str(DEFAULT_TABLE_PATH),
        help="Lookup CSV used for specular interpolation.",
    )
    return parser


def main():
    parser = _build_argument_parser()
    args = parser.parse_args()
    result = an_from_reflection_albedo(
        args.value,
        albedo_type=args.albedo_type,
        reflection_model=args.model,
        table_path=args.table_path,
    )
    print(result)


if __name__ == "__main__":
    main()
