_base_ = '../pi05_libero_official_full.py'

eval = dict(task_suite_name='libero_spatial')
themis = dict(
    evaluation=dict(
        environment=dict(task_suite_name='libero_spatial'),
        max_episode_steps=220))
