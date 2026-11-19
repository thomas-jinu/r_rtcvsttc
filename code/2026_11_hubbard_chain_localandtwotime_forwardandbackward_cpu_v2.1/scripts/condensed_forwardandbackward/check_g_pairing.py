#!/usr/bin/env python3
"""Check that each G branch pairs its P1 operator with the right measured label.

  G<: P1 = c|P0>   -> processor and batch label  <P2|c'|P3>
  G>: P1 = c'|P0>  -> processor and batch label  <P2|c|P3>

Run this after changing the G inputs, the processors, or the batch commands.
It exits with status 1 if any pair disagrees.
"""

import argparse
import importlib.util
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED = {
    "G<": {"p1": "c[{site}]*|P0>", "label": "<P2|c'|P3>", "processor": "my_hubbard_chain_twotime_G<_parallel_process.py"},
    "G>": {"p1": "c[{site}]'*|P0>", "label": "<P2|c|P3>", "processor": "my_hubbard_chain_twotime_G>_parallel_process.py"},
}


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    sys.path.insert(0, str(HERE))
    import my_hubbard_chain_td_Gless_Ggreater_singlejob_streaming as combined
    import singletime_addendum

    # Namespace with the fields evolve_input reads; values are only used to build text.
    a = type("A", (), dict(sites=4, up=2, down=2, t=-1.0, U=8.0, potentialV=0.0, finite_kept=64,
                           TSPAdvanceEach=2, TSPTau=0.5, Pump_time_steps=3, center_site=1))()
    times = [round(i * 0.5, 9) for i in range(0, 6)]
    pump = [0.0] * len(times)
    failures = []

    for branch, expected in EXPECTED.items():
        # Same flags as write_g_inputs: G< passes creator=True, G> passes creator=False,
        # and evolve_input is called with not creator.
        creator = branch == "G<"
        text = combined.evolve_input(a, "out", times, pump, 1, True, "restart", not creator)
        p1 = re.search(r'string P1="([^"]*)"', text).group(1)
        want_p1 = expected["p1"].format(site=a.center_site)
        if p1 != want_p1:
            failures.append(f"{branch}: P1 is {p1!r}, expected {want_p1!r}")

        module = load(HERE / expected["processor"], branch.replace("<", "less").replace(">", "greater"))
        if module.TWOTIME_OPERATOR_LABEL != expected["label"]:
            failures.append(f"{branch}: processor label {module.TWOTIME_OPERATOR_LABEL!r}, expected {expected['label']!r}")

        # The singletime pipeline uses the same creator convention.
        singletime_creator = branch == "G>"
        st = combined.evolve_input(a, "s", times, pump, 0, True, "restart", singletime_creator)
        st_p1 = re.search(r'string P1="([^"]*)"', st).group(1)
        if st_p1 != want_p1:
            failures.append(f"{branch}: singletime P1 is {st_p1!r}, expected {want_p1!r}")

    # The batch and post-processing commands must pass the same labels.
    batch_text = (HERE / "my_hubbard_chain_td_Gless_Ggreater_singlejob_streaming.py").read_text()
    for branch, expected in EXPECTED.items():
        if f'"{expected["label"]}"' not in batch_text and f"'{expected['label']}'" not in batch_text:
            failures.append(f"{branch}: label {expected['label']!r} not found in the combined script")

    if failures:
        print("G pairing check FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("G pairing check passed: G< uses c with <P2|c'|P3>, G> uses c' with <P2|c|P3>.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
