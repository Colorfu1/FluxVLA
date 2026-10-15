import json
import runpy
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

Recorder = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2] /
        'fluxvla/engines/runners/serving/observation_video.py')
)['ObservationVideoRecorder']


def test_video_decodes_with_both_views_and_duplicate_events(tmp_path):
    recorder = Recorder(['front', 'wrist'], fps=4, rotate_180=True)
    result = {'run_dir': str(tmp_path)}
    recorder.on_event(
        dict(request_id='run', event_type='run_start', payload={}), result)
    start = dict(
        request_id='start',
        event_type='episode_start',
        payload=dict(
            episode_id='e1',
            task_index=0,
            episode_index=0,
            description='test <task>'))
    recorder.on_event(start, result)
    front = np.zeros((32, 32, 3), np.uint8)
    front[:16, :, 0] = 255
    wrist = np.zeros_like(front)
    wrist[:, :, 2] = 255
    for _ in range(4):
        recorder.record({'front': front, 'wrist': wrist}, 'e1')
    end = dict(
        request_id='end',
        event_type='episode_end',
        payload=dict(episode_id='e1', success=True))
    recorder.on_event(end, result)
    recorder.on_event(end, result)
    recorder.close()
    output = tmp_path / 'observation_videos'
    videos = list(output.glob('*.mp4'))
    assert len(videos) == 1
    frames = imageio.mimread(videos[0])
    assert len(frames) == 4
    assert frames[0].shape == (32, 64, 3)
    assert frames[0][24, 8, 0] > 200  # Rotation: red moves to the bottom.
    assert frames[0][8, 48, 2] > 200  # Wrist view is on the right.
    assert len(json.loads((output / 'videos.json').read_text())) == 1
    assert 'test &lt;task&gt;' in (output / 'index.html').read_text()


def test_interrupted_recording_is_playable(tmp_path):
    recorder = Recorder(['front'])
    result = {'run_dir': str(tmp_path)}
    recorder.on_event(
        dict(request_id='r', event_type='run_start', payload={}), result)
    recorder.on_event(
        dict(
            request_id='e',
            event_type='episode_start',
            payload=dict(episode_id='e', task_index=0, episode_index=0)),
        result)
    recorder.record({'front': np.zeros((32, 32, 3), np.uint8)}, 'e')
    recorder.close()
    recorder.close()
    video = next((tmp_path / 'observation_videos').glob('*_interrupted.mp4'))
    assert len(imageio.mimread(video)) == 1
