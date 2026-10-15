"""Public pi05_libero weights, all tasks, 50 fixed initial states per task."""

_base_ = './pi05_libero_official_smoke.py'

eval = dict(num_trials_per_task=50, task_ids=None)
themis = dict(evaluation=dict(task_ids=None, episodes_per_task=50))
