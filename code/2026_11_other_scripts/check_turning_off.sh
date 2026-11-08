#!/usr/bin/env bash
# Report run folders whose runFor*evolve.cout* file does not contain "Turning off".
# Usage: ./check_turning_off.sh /path/to/parent_folder

set -uo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/parent_folder" >&2
    exit 2
fi

root=$1
if [[ ! -d $root ]]; then
    echo "Not a folder: $root" >&2
    exit 2
fi

checked=0
unfinished=0
while IFS= read -r -d '' cout_file; do
    checked=$((checked + 1))

    if grep -Fq -- "Turning off" "$cout_file"; then
        continue
    fi

    grep_status=$?
    if [[ $grep_status -eq 1 ]]; then
        unfinished=$((unfinished + 1))
        printf "%s — 'Turning off' was NOT found (run is not finished).\n" "$(dirname "$cout_file")"
    else
        printf "Could not read: %s\n" "$cout_file" >&2
    fi
done < <(find "$root" -type f -name 'runFor*evolve.cout*' -print0)

printf "\nChecked %d file(s); %d run(s) not finished.\n" "$checked" "$unfinished"
exit "$unfinished"
