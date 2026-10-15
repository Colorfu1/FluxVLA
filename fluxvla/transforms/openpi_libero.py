"""Normalization used by the public OpenPI LIBERO checkpoints.

OpenPI uses unclipped quantiles with epsilon=1e-6. Keep this separate from
the normalization used to train FluxVLA's existing LIBERO checkpoints.
"""

import numpy as np

from fluxvla.engines import TRANSFORMS
from .normalize import DenormalizeLiberoAction, LiberoProprioFromInputs


@TRANSFORMS.register_module()
class OpenPiLiberoProprio(LiberoProprioFromInputs):

    def _normalize_quantile(self, state, stats):
        low, high = np.asarray(stats['q01']), np.asarray(stats['q99'])
        return (state - low) / (high - low + 1e-6) * 2.0 - 1.0


@TRANSFORMS.register_module()
class OpenPiDenormalizeLiberoAction(DenormalizeLiberoAction):

    def __init__(self, norm_stats, action_dim=7):
        super().__init__(
            norm_stats=norm_stats,
            action_dim=action_dim,
            norm_type='quantile',
            normalize_gripper_action=False,
            invert_gripper_action=False)

    def __call__(self, data):
        # The legacy LIBERO transform handles individual actions. Preserve
        # the leading batch/time axes when ROS passes a whole action chunk.
        stats = self.norm_stats[data['norm_stats_key']]
        action_stats = stats.get('action')
        if action_stats is None:
            action_stats = stats['actions']
        return self._denormalize_quantile(
            np.asarray(data['action']), action_stats)

    def _denormalize_quantile(self, action, stats):
        action = action[..., :self.action_dim]
        low, high = np.asarray(stats['q01']), np.asarray(stats['q99'])
        return (action + 1.0) / 2.0 * (high - low + 1e-6) + low
