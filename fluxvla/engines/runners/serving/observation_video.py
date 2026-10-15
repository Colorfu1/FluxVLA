"""Videos of policy input observations, owned by the FluxVLA result sink."""

import html
import json
from pathlib import Path

import numpy as np


class ObservationVideoRecorder:
    """Stream camera frames at policy-request frequency, not simulator fps."""

    def __init__(self, image_keys, fps=4, rotate_180=False):
        if not image_keys or fps <= 0:
            raise ValueError(
                'Video recording needs image_keys and positive fps')
        self.image_keys = tuple(image_keys)
        self.fps = float(fps)
        self.rotate_180 = bool(rotate_180)
        self.run_dir = None
        self.episodes = {}
        self.completed = []
        self.handled_requests = set()
        self.tasks = {}

    def on_event(self, event, result):
        if event['request_id'] in self.handled_requests:
            return
        kind, payload = event['event_type'], event['payload']
        if kind == 'run_start':
            self.close()
            self.run_dir = Path(result['run_dir']) / 'observation_videos'
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.completed = []
            self.handled_requests.clear()
            self.tasks = {
                task['task_id']: task
                for task in payload.get('tasks', [])
            }
        elif kind == 'episode_start':
            episode_id = payload['episode_id']
            payload = {**self.tasks.get(payload.get('task_id'), {}), **payload}
            self.episodes.setdefault(
                episode_id, dict(payload=payload, writer=None, frames=0))
        elif kind == 'episode_end':
            self._finish(payload['episode_id'],
                         'success' if payload['success'] else 'failure')
        elif kind == 'run_end':
            self.close()
        self.handled_requests.add(event['request_id'])

    def record(self, observation, episode_id):
        episode = self.episodes.get(episode_id)
        if episode is None:
            return
        frames = []
        for key in self.image_keys:
            frame = np.asarray(observation[key])
            if frame.ndim != 3 or frame.shape[
                    2] != 3 or frame.dtype != np.uint8:
                raise ValueError(f'Video image {key} must be uint8 HWC RGB')
            frames.append(frame[::-1, ::-1] if self.rotate_180 else frame)
        frame = np.ascontiguousarray(np.concatenate(frames, axis=1))
        if episode['writer'] is None:
            import imageio.v2 as imageio
            payload = episode['payload']
            stem = (f"task{int(payload['task_index'])}"
                    f"_episode{int(payload['episode_index'])}")
            episode['path'] = self.run_dir / f'{stem}.partial.mp4'
            episode['writer'] = imageio.get_writer(
                str(episode['path']),
                fps=self.fps,
                codec='libx264',
                quality=7,
                macro_block_size=1,
                ffmpeg_log_level='error')
        episode['writer'].append_data(frame)
        episode['frames'] += 1

    def _finish(self, episode_id, status):
        episode = self.episodes.get(episode_id)
        if episode is None:
            return
        if episode['writer'] is not None:
            episode['writer'].close()
            path = episode['path']
            target = path.with_name(
                path.name.replace('.partial', f'_{status}'))
            path.replace(target)
            self.completed.append(
                dict(
                    file=target.name,
                    status=status,
                    frames=episode['frames'],
                    fps=self.fps,
                    image_keys=self.image_keys,
                    description=episode['payload'].get('description', ''),
                    sampling='one frame per policy request'))
            self._write_index()
        del self.episodes[episode_id]

    def _write_index(self):
        (self.run_dir / 'videos.json').write_text(
            json.dumps(self.completed, indent=2), encoding='utf-8')
        cards = ''.join(
            f'<h2>{html.escape(item["description"])} — {item["status"]}</h2>'
            f'<video controls width="768" src="{item["file"]}"></video>'
            for item in self.completed)
        (self.run_dir / 'index.html').write_text(
            '<!doctype html><meta charset="utf-8">'
            '<title>LIBERO smoke videos</title>'
            '<style>body{font:16px sans-serif;max-width:900px;'
            'margin:40px auto;'
            'background:#16191e;color:#eee}video{max-width:100%}</style>'
            '<h1>LIBERO evaluation</h1>'
            '<p>Left: agent view. Right: wrist view. '
            'Frames are sampled at policy requests '
            '(one frame per replan).</p>' + cards,
            encoding='utf-8')

    def close(self):
        for episode_id in list(self.episodes):
            self._finish(episode_id, 'interrupted')
