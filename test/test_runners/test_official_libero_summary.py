import json
import runpy
from pathlib import Path

import pytest

runtime = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2] /
        'tools/run_pi05_official_libero.py'))


def test_four_suite_average_excludes_libero90_and_tracks_completion():
    suites = list(runtime['SUITES'])
    rows = [
        dict(suite=suite, episodes=500, success_rate_pct=score)
        for suite, score in zip(suites[:4], (100, 80, 60, 40))
    ]
    report = runtime['make_report'](rows[:3], suites)
    assert report['libero_four_suite_average_success_rate_pct'] is None
    assert not report['standard_test_complete']
    report = runtime['make_report'](rows, suites)
    assert not report['complete']
    assert report['standard_test_complete']
    assert report['libero_four_suite_average_success_rate_pct'] == 70
    rows.append(dict(suite='libero_90', episodes=4500, success_rate_pct=10))
    report = runtime['make_report'](rows, suites)
    assert report['complete']
    assert report['completed_episodes'] == report['expected_episodes'] == 6500
    assert report['libero_four_suite_average_success_rate_pct'] == 70


def test_summary_recomputes_scores_and_rejects_partial_task_coverage(tmp_path):
    tasks = {
        f'libero_10_{i}': dict(
            total_episodes=50,
            successes=i + 40,
            success_rate=-1,
            task_description=str(i))
        for i in range(10)
    }
    data = dict(
        task_results=tasks,
        suite_stats=dict(
            libero_10=dict(
                total_tasks=10,
                total_trials=500,
                total_successes=445,
                total_time=1000)))
    path = tmp_path / 'summary.json'
    path.write_text(json.dumps(data))
    row, details = runtime['read_suite_metrics'](path, 'libero_10')
    assert row['success_rate_pct'] == 89
    assert row['mean_episode_time_s'] == 2
    assert details[0]['success_rate_pct'] == 80
    tasks['libero_10_9']['total_episodes'] = 49
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='incomplete'):
        runtime['read_suite_metrics'](path, 'libero_10')
    del tasks['libero_10_9']
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='incomplete'):
        runtime['read_suite_metrics'](path, 'libero_10')
