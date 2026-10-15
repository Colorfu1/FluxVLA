"""Guard against legacy overrides silently falling back to LIBERO-10."""

from pathlib import Path

import pytest
from mmengine import Config

ROOT = Path(__file__).resolve().parents[2] / 'configs/pi05'


@pytest.mark.parametrize('suite,steps', [('libero_spatial', 220),
                                         ('libero_object', 280),
                                         ('libero_goal', 300),
                                         ('libero_10', 520),
                                         ('libero_90', 400)])
def test_official_suite_uses_zmq_and_original_protocol(suite, steps):
    cfg = Config.fromfile(str(ROOT / 'libero_official' / f'{suite}.py'))
    assert 'runner' not in cfg.themis and 'ros_server' not in cfg.themis
    evaluation = cfg.themis.evaluation
    assert evaluation.environment.task_suite_name == suite
    assert cfg.eval.task_suite_name == suite
    assert evaluation.max_episode_steps == steps
    assert evaluation.execute_horizon == 5
    assert evaluation.episodes_per_task == cfg.eval.num_trials_per_task == 50
    assert evaluation.task_ids is None and cfg.eval.task_ids is None
    assert evaluation.strict_worker_assignment
    assert cfg.themis.server.strict_worker_assignment
    assert cfg.themis.server.keep_params_fp32
    assert cfg.themis.server.runner.sequence_noise
    assert cfg.model.openpi_fp32_flow
    assert cfg.model.vision_backbone.openpi_stem_fp32
    assert cfg.eval.denormalize_action.type == 'OpenPiDenormalizeLiberoAction'
    assert cfg.themis.transport.unnorm_key == 'physical-intelligence/libero'


def test_smoke_limits_apply_to_new_evaluation_section():
    cfg = Config.fromfile(str(ROOT / 'pi05_libero_official_smoke.py'))
    assert cfg.themis.evaluation.task_ids == ['0']
    assert cfg.themis.evaluation.episodes_per_task == 1
    assert cfg.eval.num_trials_per_task == 1
