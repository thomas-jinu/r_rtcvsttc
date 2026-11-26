"""Postprocess existing TD/G or XAS output without running DMRG++."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def dispatch():
    """Handle the exclusive --postprocess-only PATH mode before normal parsing."""
    if '--postprocess-only' not in sys.argv[1:]:
        return False
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--postprocess-only', required=True, type=Path,
                        help='Existing TD directory, or XAS manifest.json')
    path = parser.parse_args().postprocess_only.resolve()
    scripts = Path(__file__).resolve().parent
    if path.is_file():
        subprocess.run([sys.executable, str(scripts / 'hubbard_chain' /
                        'my_hubbard_chain_td_XAS_staged.py'), '--aggregate', str(path)], check=True)
        return True
    if not path.is_dir():
        parser.error(f'Not a directory or manifest: {path}')
    args_files = list(path.glob('input_args_td_*.json'))
    logs = list(path.glob('runForinput_td_*.cout'))
    if len(args_files) != 1 or len(logs) != 1:
        parser.error('Expected exactly one input_args_td_*.json and one runForinput_td_*.cout')
    args = json.loads(args_files[0].read_text())
    pump = path / 'pump_table.csv'
    post = scripts / 'post_process'
    commands = [[sys.executable, str(post / 'td_process.py'), str(logs[0]), str(pump)]]
    branches = [path.parent / f'{branch}_twotime_{path.name}' for branch in ('G<', 'G>')]
    if any(folder.exists() for folder in branches):
        if not all(folder.is_dir() for folder in branches):
            parser.error('Incomplete G branch directories: both G< and G> are required')
        for folder, token in zip(branches, ('gless', 'ggreater')):
            commands.append([sys.executable, str(post / f'twotime_{token}_process.py'),
                             str(args['center_site']), str(path.parent / Path(args['gs_filename']).name),
                             str(pump), str(folder), '--workers', '1'])
        # Ground-state files live beside TD folders; infer their current location
        # so relative original arguments and moved run trees remain usable.
    for command in commands:
        subprocess.run(command, check=True)
    return True
