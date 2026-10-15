"""Public gs://openpi-assets/checkpoints/pi05_libero, single-GPU smoke.

No training dataset is needed. Use tools/convert_pi05_official.py to prepare
the checkpoint and its original normalization statistics.
"""

_base_ = './pi05_paligemma_libero_10_full_finetune.py'
_assets = '{{fileDirname}}/../../checkpoints/pi05_libero'

model = dict(
    pretrained_name_or_path=None,
    openpi_fp32_flow=True,
    vision_backbone=dict(openpi_stem_fp32=True),
    llm_expert=dict(use_token_embedding=False),
    attention_implementation='sdpa',
    n_action_steps=10,
    num_steps=10,
)

eval = dict(
    model_family='pi05_official',
    norm_stats_key='physical-intelligence/libero',
    num_trials_per_task=1,
    eval_chunk_size=5,
    dataset=dict(
        num_padding_imgs=1,
        transforms=[
            dict(
                type='ProcessLiberoEvalInputs',
                img_keys=['agentview_image', 'robot0_eye_in_hand_image']),
            dict(
                type='TransformImage',
                image_resize_strategy='resize-naive',
                input_sizes=[[3, 224, 224], [3, 224, 224]],
                means=[[127.5] * 3] * 2,
                stds=[[127.5] * 3] * 2),
            dict(
                type='LiberoPromptFromInputs',
                use_conversation=False,
                max_len=200,
                tokenizer=dict(
                    type='PaligemmaTokenizer',
                    model_path=_assets + '/tokenizer')),
            dict(
                type='OpenPiLiberoProprio', norm_type='quantile',
                state_dim=32),
        ]),
    denormalize_action=dict(
        _delete_=True, type='OpenPiDenormalizeLiberoAction', action_dim=7),
    # Videos show the observations received at each policy replan (every 5
    # simulation steps), at 20 / 5 = 4 fps. Both camera views are included.
    observation_video=dict(
        image_keys=['agentview_image', 'robot0_eye_in_hand_image'],
        rotate_180=True,
        fps=4),
)

themis = dict(
    transport=dict(unnorm_key='physical-intelligence/libero'),
    evaluation=dict(
        task_ids=['0'],
        episodes_per_task=1,
        execute_horizon=5,
        max_episode_steps=520,
        strict_worker_assignment=True),
    server=dict(
        strict_worker_assignment=True,
        # Preserve the official FP32 flow projections and selective BF16
        # parameters established by the model constructor.
        keep_params_fp32=True,
        runner=dict(sequence_noise=True),
        ckpt_path=_assets + '/model.safetensors',
        norm_stats_path=_assets + '/dataset_statistics.json',
        evaluation_reporting=dict(
            result_output_dir='{{fileDirname}}/../../work_dirs/fluxthemis'),
    ),
)
