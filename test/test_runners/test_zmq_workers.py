"""Shared-GPU startup and fixed pairing on the stateless ZMQ service."""

import os
import runpy
from pathlib import Path

import pytest

from fluxvla.engines.runners.serving.distributed_server import \
    FluxVLAInferenceServer
from fluxvla.engines.runners.serving.zmq_protocol import empty_payload

ROOT = Path(__file__).resolve().parents[2]
entrypoint = runpy.run_path(str(ROOT / 'scripts/distributed_zmq_server.py'))
model_device = entrypoint['model_device']


def make_server(monkeypatch, count=4, **kwargs):
    server = FluxVLAInferenceServer(
        'inproc://front', 'inproc://back', expected_workers=count, **kwargs)
    sent = []
    monkeypatch.setattr(
        server, '_send_worker', lambda kind, worker, **values: sent.append(
            (kind, worker.worker_id, values)))
    return server, sent


def register(server, index, device, node='host'):
    server._register_worker(
        str(index).encode(),
        dict(worker_id=f'model-worker-{index}', device=device, node_id=node))
    return server._workers[str(index).encode()]


def test_rank_mapping_keeps_unique_gpu_visibility():
    assert [model_device(i, 4, 2, 2)
            for i in range(4)] == ['cuda:0', 'cuda:1', 'cuda:0', 'cuda:1']
    assert [model_device(i, 2, 2, 1) for i in range(2)] == ['cuda:0'] * 2
    with pytest.raises(ValueError):
        model_device(0, 4, 1, 2)
    with pytest.raises(ValueError):
        model_device(0, 3, 2, 2)


@pytest.mark.parametrize('rank,local_rank,visible,replicas', [
    (14, 14, '0,1,2,3,4,5,6,7', 3),
    (23, 23, '0,1,2,3,4,5,6,7', 3),
    (38, 14, '0,1,2,3,4,5,6,7', 3),
    (3, 3, '3,5', 2),
    (3, 3, 'GPU-first,GPU-second', 2),
    (1, 1, '3,5', 1),
])
def test_entrypoint_initializes_mmengine_logger_for_shared_gpu(
        tmp_path, monkeypatch, rank, local_rank, visible, replicas):
    """Exercise MMEngine's actual CUDA logging path without CUDA hardware."""
    import torch
    from mmengine import Config
    from mmengine.logging import MMLogger
    from mmengine.logging.logger import _get_device_id

    from fluxvla.engines.runners.serving import distributed_server as module

    tokens = visible.split(',')
    local_world_size = len(tokens) * replicas
    world_size = local_world_size * (rank // local_world_size + 1)
    for name, value in dict(
            RANK=rank,
            LOCAL_RANK=local_rank,
            WORLD_SIZE=world_size,
            LOCAL_WORLD_SIZE=local_world_size,
            CUDA_VISIBLE_DEVICES=visible).items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: len(tokens))
    device_index = local_rank % len(tokens)
    physical_device = tokens[device_index]
    expected_device = (
        int(physical_device)
        if physical_device.isdecimal() else physical_device)
    load_config = Config.fromfile
    logger_devices = []

    def checked_config(*args, **kwargs):
        # Custom config imports may initialize MMLogger before model loading.
        logger = MMLogger(f'config-rank-{rank}')
        logger.info('shared-GPU config initialization')
        logger_devices.append(_get_device_id())
        return load_config(*args, **kwargs)

    def launch(**kwargs):
        logger = MMLogger(f'model-rank-{rank}')
        logger.info('shared-GPU model initialization')
        logger_devices.append(_get_device_id())
        assert kwargs['rank'] == rank
        assert kwargs['world_size'] == world_size
        assert kwargs['device'] == f'cuda:{device_index}'
        assert os.environ['RANK'] == str(rank)
        assert os.environ['WORLD_SIZE'] == str(world_size)
        assert os.environ['LOCAL_WORLD_SIZE'] == str(local_world_size)
        assert os.environ['CUDA_VISIBLE_DEVICES'] == visible

    monkeypatch.setattr(Config, 'fromfile', checked_config)
    monkeypatch.setattr(module, 'launch_server_task', launch)
    config = tmp_path / 'model.py'
    config.write_text('themis = dict(server=dict())\neval = dict()\n')
    checkpoint = tmp_path / 'model.safetensors'
    checkpoint.touch()
    assert entrypoint['main']([
        '--config',
        str(config), '--ckpt-path',
        str(checkpoint), '--workers-per-gpu',
        str(replicas)
    ]) == 0
    assert logger_devices == [expected_device, expected_device]


def test_loads_parallel_devices_serial_replicas_and_preserves_rank(
        monkeypatch):
    server, sent = make_server(monkeypatch)
    # Registration order cannot change rank order or overlap GPU loaders.
    workers = {i: register(server, i, f'cuda:{i % 2}') for i in (3, 2, 1, 0)}
    server._schedule_loads()
    assert [(kind, worker)
            for kind, worker, _ in sent] == [('load_worker', 'model-worker-0'),
                                             ('load_worker', 'model-worker-1')]
    server._mark_worker_ready(workers[1])
    server._schedule_loads()
    assert sent[-1][:2] == ('load_worker', 'model-worker-3')
    assert workers[2].loading_started_at is None
    server._mark_worker_ready(workers[0])
    server._schedule_loads()
    assert sent[-1][:2] == ('load_worker', 'model-worker-2')
    assert len(sent) == 4
    server._schedule_loads()
    assert len(sent) == 4


def test_loading_cap_and_node_local_device_identity(monkeypatch):
    server, sent = make_server(monkeypatch, count=2, startup_parallelism=1)
    first = register(server, 0, 'cuda:0', 'node-a')
    register(server, 1, 'cuda:0', 'node-b')
    server._schedule_loads()
    assert len(sent) == 1
    server._mark_worker_ready(first)
    server._schedule_loads()
    assert len(sent) == 2
    server, sent = make_server(monkeypatch, count=2)
    register(server, 0, 'cuda:0', 'node-a')
    register(server, 1, 'cuda:0', 'node-b')
    server._schedule_loads()
    assert len(sent) == 2


def test_fixed_pairing_never_steals_a_busy_shard(monkeypatch):
    server, sent = make_server(
        monkeypatch, count=2, strict_worker_assignment=True)
    workers = [register(server, i, 'cuda:0') for i in range(2)]
    for worker in reversed(workers):
        server._mark_worker_ready(worker)
    for ordinal in (1, 3, 2):
        server._queue_prediction((b'client', ),
                                 dict(
                                     request_id=str(ordinal),
                                     episode_id=f'session:{ordinal}',
                                     policy_seed=ordinal,
                                     sequence=2), empty_payload())
    server._dispatch_predictions()
    assert [(worker, args['request_id'])
            for _, worker, args in sent] == [('model-worker-0', '1'),
                                             ('model-worker-1', '2')]
    assert len(server._pending_predictions) == 1
    assert sent[0][2]['sequence'] == 2
    server._release_worker(workers[1])
    server._dispatch_predictions()
    assert len(sent) == 2
    server._release_worker(workers[0])
    server._dispatch_predictions()
    assert sent[-1][1:] == ('model-worker-0',
                            dict(
                                request_id='3',
                                policy_seed=3,
                                sequence=2,
                                unnorm_key='',
                                payload=empty_payload()))


def test_dynamic_routing_remains_available_for_robodojo(monkeypatch):
    server, sent = make_server(monkeypatch, count=2)
    register(server, 0, 'cuda:0')
    ready = register(server, 1, 'cuda:1')
    server._mark_worker_ready(ready)
    server._queue_prediction((b'client', ),
                             dict(request_id='r', policy_seed=7),
                             empty_payload())
    server._dispatch_predictions()
    assert sent[0][1] == 'model-worker-1'


def test_real_zmq_lifecycle_with_two_shared_device_workers(
        tmp_path, monkeypatch):
    """Check load grants, fixed routing and shutdown over real sockets."""
    import threading
    import time

    import numpy as np
    import zmq

    from fluxvla.engines.runners.serving import distributed_server as module
    from fluxvla.engines.runners.serving.zmq_protocol import (decode_frame,
                                                              decode_header,
                                                              encode_frame,
                                                              encode_header)

    config = tmp_path / 'model.py'
    config.write_text('themis = {}\n')
    frontend = f'ipc://{tmp_path}/front.sock'
    backend = f'ipc://{tmp_path}/back.sock'
    failures = []
    loaded = []

    class Runner:

        def __init__(self):
            self.rank = int(threading.current_thread().name.rsplit('-', 1)[1])
            loaded.append(self.rank)

        def predict_action(self, observation, unnorm_key, policy_seed,
                           sequence):
            assert observation['image'].shape == (4, 4, 3)
            return np.array([[self.rank, policy_seed, sequence]],
                            np.float32), 0.01

    class Reporter:

        def process_event(self, **kwargs):
            return {'accepted': True, 'run_dir': str(tmp_path)}

    monkeypatch.setattr(module, 'build_serving_runner_from_config',
                        lambda *args, **kwargs: Runner())
    server = module.FluxVLAInferenceServer(
        frontend,
        backend,
        Reporter(),
        expected_workers=2,
        strict_worker_assignment=True,
        startup_timeout_s=10,
        deployment_id='test-deployment')

    def guarded(function, **kwargs):
        try:
            function(**kwargs)
        except BaseException as exc:
            failures.append(exc)

    threads = [
        threading.Thread(target=guarded, args=(server.run, ), daemon=True)
    ]
    for rank in (1, 0):
        threads.append(
            threading.Thread(
                target=guarded,
                args=(module.run_model_worker, ),
                kwargs=dict(
                    config_path=str(config),
                    ckpt_path='unused',
                    backend_endpoint=backend,
                    worker_id=f'model-worker-{rank}',
                    node_id='host',
                    deployment_id='test-deployment',
                    device='cuda:0'),
                name=f'model-worker-{rank}',
                daemon=True))
    context = zmq.Context()
    client = context.socket(zmq.REQ)
    client.setsockopt(zmq.RCVTIMEO, 10000)
    client.setsockopt(zmq.LINGER, 0)
    client.connect(frontend)

    def request(kind, payload, **headers):
        client.send_multipart([
            encode_header(
                kind,
                request_id=kind,
                deployment_id='test-deployment',
                **headers),
            encode_frame(payload)
        ])
        frames = client.recv_multipart()
        header = decode_header(frames[0])
        assert header['ok'], header
        return decode_frame(frames[1])

    try:
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 10
        while request('status', {}).get('status') != 'ready':
            assert time.monotonic() < deadline, failures
            time.sleep(0.01)
        assert loaded == [0, 1]
        request(
            'report_event', {
                'parallel_workers': 2,
                'strict_worker_assignment': True
            },
            event_type='run_start',
            run_session_id='session',
            sequence=1)
        for ordinal in (1, 2, 3, 4):
            result = request(
                'predict_action', {'image': np.zeros((4, 4, 3), np.uint8)},
                episode_id=f'session:{ordinal}',
                policy_seed=7 + ordinal,
                sequence=2)
            assert result['actions'].tolist() == [[(ordinal - 1) % 2,
                                                   7 + ordinal, 2]]
        request(
            'report_event', {},
            event_type='run_end',
            run_session_id='session',
            sequence=2)
        for thread in threads:
            thread.join(timeout=5)
        assert not any(thread.is_alive() for thread in threads)
        assert not failures
    finally:
        client.close()
        context.term()
