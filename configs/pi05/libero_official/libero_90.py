_base_ = '../pi05_libero_official_full.py'

eval = dict(task_suite_name='libero_90')
themis = dict(
    evaluation=dict(
        environment=dict(task_suite_name='libero_90'), max_episode_steps=400))
