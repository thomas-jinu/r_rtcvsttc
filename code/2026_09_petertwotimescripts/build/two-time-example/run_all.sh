#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
: "${DMRG:?Set DMRG to the absolute path of the dmrg executable}"
DMRG_ALLOW_VALIDATED_ABORT=${DMRG_ALLOW_VALIDATED_ABORT:-1}

validate_completed_abort() {
    checkpoint=$1
    log=$2
    test -s "$checkpoint" || return 1
    test -s "$log" || return 1
    grep -Eq 'Checkpoint .*: Written sys\. and env\. stacks to disk\.$' "$log" || return 1
    if command -v h5dump >/dev/null 2>&1; then
        h5dump -H "$checkpoint" >/dev/null 2>&1 || return 1
    fi
}

run() {
    directory=$1
    input=$2
    checkpoint=$ROOT/$3
    log=$ROOT/$4
    shift 4
    echo "==> $directory/$input"

    # Never mistake products from an earlier invocation for successful output.
    rm -f "$checkpoint" "$log"
    set +e
    (cd "$ROOT/$directory" && "$DMRG" -f "$input" "$@")
    status=$?
    set -e
    if [ "$status" -eq 0 ]; then
        return 0
    fi

    if [ "$status" -eq 134 ] && [ "$DMRG_ALLOW_VALIDATED_ABORT" = 1 ] \
        && validate_completed_abort "$checkpoint" "$log"; then
        echo "WARNING: validated DMRG++ shutdown abort for $directory/$input; continuing." >&2
        return 0
    fi

    echo "ERROR: DMRG++ failed for $directory/$input (exit $status)." >&2
    return "$status"
}

run gs input.inp gs/data.hd5 gs/runForinput.cout
run column-t000 insert.inp column-t000/insert.hd5 column-t000/runForinsert.cout '<P0|P0>'
run column-t000 evolve.inp column-t000/evolve.hd5 column-t000/runForevolve.cout '<P2|c|P1>,<P2.last|P2>'
run reference-t001 input.inp reference-t001/data.hd5 reference-t001/runForinput.cout '<P0|P0>'
run column-t001 insert.inp column-t001/insert.hd5 column-t001/runForinsert.cout '<P0|P0>,<P1|P1>'
run column-t001 evolve.inp column-t001/evolve.hd5 column-t001/runForevolve.cout '<P2|c|P3>,<P2.last|P2>'
run reference-t002 input.inp reference-t002/data.hd5 reference-t002/runForinput.cout '<P0|P0>'
run column-t002 insert.inp column-t002/insert.hd5 column-t002/runForinsert.cout '<P0|P0>,<P1|P1>'
run column-t002 evolve.inp column-t002/evolve.hd5 column-t002/runForevolve.cout '<P2|c|P3>,<P2.last|P2>'
run reference-t003 input.inp reference-t003/data.hd5 reference-t003/runForinput.cout '<P0|P0>'
run column-t003 insert.inp column-t003/insert.hd5 column-t003/runForinsert.cout '<P0|P0>,<P1|P1>'
run column-t003 evolve.inp column-t003/evolve.hd5 column-t003/runForevolve.cout '<P2|c|P3>,<P2.last|P2>'
run reference-t004 input.inp reference-t004/data.hd5 reference-t004/runForinput.cout '<P0|P0>'
run column-t004 insert.inp column-t004/insert.hd5 column-t004/runForinsert.cout '<P0|P0>,<P1|P1>'
run column-t004 evolve.inp column-t004/evolve.hd5 column-t004/runForevolve.cout '<P2|c|P3>,<P2.last|P2>'
