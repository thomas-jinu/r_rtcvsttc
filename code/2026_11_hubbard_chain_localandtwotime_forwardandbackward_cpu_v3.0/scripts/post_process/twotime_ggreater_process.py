"""G> v2.4: retain last forward equal-time values and report sweep disagreement."""
from twotime_postprocess import (
    COLLECTION_MARKER as TWOTIME_COLLECTION_MARKER,
    OVERLAP_TOLERANCE,
    NUMBER as TWOTIME_NUMBER_PATTERN,
    main as shared_main,
    operator_pattern,
    process_twotime_results as shared_process,
    read_operator_output as shared_read,
    read_pump_file,
    save_twotime_csv,
)

TWOTIME_OPERATOR_LABEL = "<P2|c|P3>"


def read_operator_output(output_path, operator_label=TWOTIME_OPERATOR_LABEL, equal_time=None):
    return shared_read(output_path, operator_label, equal_time)


def process_twotime_results(base_directory, run_name, number_of_steps, time_axis, center_site, workers, sites=None):
    return shared_process(base_directory, run_name, number_of_steps, time_axis,
        center_site, workers, TWOTIME_OPERATOR_LABEL, "P2_c_P3", sites)


def main():
    return shared_main(TWOTIME_OPERATOR_LABEL, "P2_c_P3")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
