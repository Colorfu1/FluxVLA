import numpy as np

from fluxvla.transforms.openpi_libero import (OpenPiDenormalizeLiberoAction,
                                              OpenPiLiberoProprio)


def test_public_checkpoint_preserves_quantile_outliers_and_gripper():
    stats = {'q01': [-1.] * 7, 'q99': [1.] * 7}
    transform = OpenPiDenormalizeLiberoAction(
        norm_stats={'official': {
            'action': stats
        }})
    normalized = np.full((10, 32), 0.25)
    normalized[0, 0] = 2.0
    normalized[0, 6] = -0.4
    action = transform({'action': normalized, 'norm_stats_key': 'official'})
    assert action.shape == (10, 7)
    assert action[0, 0] > 1.0  # OpenPI does not clip quantile outliers.
    np.testing.assert_allclose(action[0, 6], -0.4, atol=1e-6)
    np.testing.assert_allclose(action[1, 6], 0.25, atol=1e-6)


def test_public_state_normalization_retains_outliers():
    transform = OpenPiLiberoProprio(norm_type='quantile', state_dim=32)
    result = transform({
        'robot0_eef_pos': np.array([3., 0., 0.]),
        'robot0_eef_quat': np.array([0., 0., 0., 1.]),
        'robot0_gripper_qpos': np.array([0.02, -0.02]),
        'norm_stats': {
            'proprio': {
                'q01': [-1.] * 8,
                'q99': [1.] * 8
            }
        },
    })['states']
    assert result.shape == (32, )
    assert result[0] > 2.99
    np.testing.assert_array_equal(result[8:], 0)
