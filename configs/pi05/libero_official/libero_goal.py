_base_ = '../pi05_libero_official_full.py'

eval = dict(task_suite_name='libero_goal')
themis = dict(
    evaluation=dict(
        environment=dict(
            task_suite_name='libero_goal'), max_episode_steps=300))
