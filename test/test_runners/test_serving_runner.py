import torch

from fluxvla.engines.runners.serving.serving_runner import FluxVLAServingRunner


class _Policy:

    def __init__(self):
        self.seeds = []
        self.device = None
        self.dtype = None

    def eval(self):
        return self

    def to(self, device, dtype=None):
        self.device = device
        self.dtype = dtype
        return self

    def predict_action(self, **batch):
        self.seeds.append(batch['seed'])
        return torch.zeros(2, 3)


def test_serving_runner_receives_episode_seed():
    model = _Policy()
    runner = FluxVLAServingRunner(
        model,
        dataset=lambda observation: {},
        device='cpu',
        enable_mixed_precision=True,
        model_outputs_environment_actions=True,
    )

    actions, _ = runner.predict_action({}, '', policy_seed=17)

    assert actions.shape == (2, 3)
    assert model.seeds == [17]
    assert model.dtype == torch.bfloat16
    assert model.device == torch.device('cpu')


def test_serving_runner_can_keep_parameters_fp32():
    model = _Policy()

    FluxVLAServingRunner(
        model,
        dataset=lambda observation: {},
        device='cpu',
        enable_mixed_precision=True,
        keep_params_fp32=True,
        model_outputs_environment_actions=True,
    )

    assert model.dtype is None
    assert model.device == torch.device('cpu')


def test_pi05_stateless_noise_matches_episode_rng_stream():

    class NoisePolicy(_Policy):
        n_action_steps = 10
        max_action_dim = 32

        def sample_noise(self, shape, device):
            return torch.normal(0., 1., size=shape, device=device)

        def predict_action(self, **batch):
            return batch['noise'][0]

    torch.manual_seed(7)
    expected = [torch.normal(0., 1., size=(1, 10, 32))[0] for _ in range(3)]
    runner = FluxVLAServingRunner(
        NoisePolicy(),
        dataset=lambda observation: {'states': torch.zeros(1, 7)},
        device='cpu',
        sequence_noise=True,
        keep_params_fp32=True,
        model_outputs_environment_actions=True)
    # Reordering and retrying requests must not change any action noise.
    for sequence in (3, 1, 2, 3):
        actions, _ = runner.predict_action({},
                                           '',
                                           policy_seed=7,
                                           sequence=sequence)
        torch.testing.assert_close(
            torch.from_numpy(actions), expected[sequence - 1], rtol=0, atol=0)
