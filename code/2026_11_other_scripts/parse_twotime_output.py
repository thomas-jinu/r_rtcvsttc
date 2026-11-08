"""Parse DMRG++ two-time operator lines of the form:

    4 0.000100122903 0.000000000000 <gs|c'|P2> 0.000000000000

The fields are site, real part, imaginary part, operator label, and time.
"""
import re
from pathlib import Path

TWOTIME_OPERATOR_LABEL = "<gs|c'|P2>"
TWOTIME_COLLECTION_MARKER = "FiniteLoops printing ends"
TWOTIME_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
TWOTIME_OPERATOR_PATTERN = re.compile(
    rf"^\s*(\d+)\s+"
    rf"({TWOTIME_NUMBER_PATTERN})\s+"
    rf"({TWOTIME_NUMBER_PATTERN})\s+"
    rf"{re.escape(TWOTIME_OPERATOR_LABEL)}\s+"
    rf"({TWOTIME_NUMBER_PATTERN})\s*$"
)


def read_operator_output(output_path: Path) -> dict[tuple[int, float], complex]:
    """Return {(site_index, time): complex_value}; last occurrence wins."""
    values: dict[tuple[int, float], complex] = {}
    collecting = False

    with Path(output_path).open("r", encoding="utf-8") as file:
        for line in file:
            if TWOTIME_COLLECTION_MARKER in line:
                collecting = True
                continue
            if not collecting or TWOTIME_OPERATOR_LABEL not in line:
                continue

            match = TWOTIME_OPERATOR_PATTERN.search(line)
            if match is None:
                continue

            site_index = int(match.group(1))
            real_value = float(match.group(2))
            imaginary_value = float(match.group(3))
            time = float(match.group(4))
            values[(site_index, time)] = complex(real_value, imaginary_value)

    return values
