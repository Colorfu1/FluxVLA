"""Run the standard four-suite LIBERO test and collect its metrics."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

SUITES = {
    'libero_spatial': (10, 220),
    'libero_object': (10, 280),
    'libero_goal': (10, 300),
    'libero_10': (10, 520),
    'libero_90': (90, 400),
}
FOUR_SUITES = tuple(SUITES)[:4]
TRIALS = 50


def read_suite_metrics(path, suite):
    """Reject partial runs instead of reporting them as full-suite scores."""
    data = json.loads(Path(path).read_text())
    stats = data['suite_stats'][suite]
    tasks = data['task_results']
    expected = {f'{suite}_{i}' for i in range(SUITES[suite][0])}
    if set(tasks) != expected or any(item['total_episodes'] != TRIALS
                                     for item in tasks.values()):
        raise ValueError(
            f'{suite}: incomplete task/episode coverage in {path}')
    episodes = sum(item['total_episodes'] for item in tasks.values())
    successes = sum(item['successes'] for item in tasks.values())
    if (stats['total_trials'] != episodes
            or stats['total_successes'] != successes
            or stats['total_tasks'] != len(expected)
            or any(not 0 <= item['successes'] <= TRIALS
                   for item in tasks.values())):
        raise ValueError(f'{suite}: inconsistent metrics in {path}')
    row = dict(
        suite=suite,
        tasks=len(tasks),
        episodes=episodes,
        successes=successes,
        success_rate_pct=100.0 * successes / episodes,
        total_time_s=stats['total_time'],
        mean_episode_time_s=stats['total_time'] / episodes,
        result_dir=str(Path(path).parent))
    task_rows = [
        dict(
            suite=suite,
            task_id=i,
            episodes=tasks[f'{suite}_{i}']['total_episodes'],
            successes=tasks[f'{suite}_{i}']['successes'],
            success_rate_pct=100.0 * tasks[f'{suite}_{i}']['successes'] /
            TRIALS,
            description=tasks[f'{suite}_{i}']['task_description'])
        for i in range(SUITES[suite][0])
    ]
    return row, task_rows


def make_report(rows, requested):
    by_suite = {row['suite']: row for row in rows}
    four_average = None
    if all(suite in by_suite for suite in FOUR_SUITES):
        four_average = sum(by_suite[s]['success_rate_pct']
                           for s in FOUR_SUITES) / len(FOUR_SUITES)
    return dict(
        complete=set(by_suite) == set(requested),
        standard_test_complete=all(suite in by_suite for suite in FOUR_SUITES),
        requested_suites=requested,
        episodes_per_task=TRIALS,
        expected_episodes=sum(SUITES[s][0] * TRIALS for s in requested),
        completed_episodes=sum(row['episodes'] for row in rows),
        suites=rows,
        libero_four_suite_average_success_rate_pct=four_average,
        four_suite_average_members=list(FOUR_SUITES),
        note='LIBERO-90 is reported separately from the four-suite average.')


def write_results(directory, rows, task_rows, requested):
    report = make_report(rows, requested)
    temporary = directory / 'summary.json.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(directory / 'summary.json')
    for name, records in (('summary.csv', rows), ('task_success_rates.csv',
                                                  task_rows)):
        if records:
            with (directory / name).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--themis-root', type=Path, required=True)
    parser.add_argument(
        '--suites',
        nargs='+',
        default=['standard'],
        choices=['standard', 'all', *SUITES],
        help='Default: standard four-suite test; all also includes LIBERO-90.')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configs without starting models or episodes.')
    args = parser.parse_args()
    launcher = args.themis_root.expanduser().resolve(
    ) / 'scripts/run_libero.py'
    if not launcher.is_file():
        parser.error('The FluxThemis launcher was not found.')
    command = [
        sys.executable,
        str(launcher), '--mode', 'eval', '--gpus', '1', '--suites',
        *args.suites
    ]
    if args.dry_run:
        command.append('--dry-run')
    raise SystemExit(subprocess.call(command))


if __name__ == '__main__':
    main()
