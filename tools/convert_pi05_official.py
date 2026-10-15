"""Convert the public pi05_libero Orbax checkpoint without a training install.

Run in an isolated CPU environment with torch, numpy==1.26.4,
jax[cpu]==0.4.35, orbax-checkpoint==0.11.5 and safetensors.
The two tensor mapping functions are read from a pinned official OpenPI
checkout; the resulting checkpoint is validated by FluxVLA's strict loader.
"""

import argparse
import ast
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import orbax.checkpoint as ocp
import torch
from safetensors.torch import save_file

OPENPI_COMMIT = '215abfb217dbac7d5f1273282331b9b1866c0479'


def flatten(tree, prefix=()):
    result = {}
    for key, value in tree.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            result.update(flatten(value, path))
        else:
            result[path] = value
    return result


def main():
    jax.config.update('jax_platforms', 'cpu')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--openpi-root', type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ['git', '-C', str(args.openpi_root), 'rev-parse', 'HEAD'],
        text=True).strip()
    if commit != OPENPI_COMMIT:
        raise ValueError(f'Expected OpenPI {OPENPI_COMMIT}, got {commit}')
    upstream = args.openpi_root / 'examples/convert_jax_model_to_pytorch.py'
    source = upstream.read_text()
    names = {'slice_paligemma_state_dict', 'slice_gemma_state_dict'}
    functions = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(functions) == 2
    namespace = {'np': np, 'torch': torch}
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]), str(upstream),
            'exec'), namespace)

    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'model.safetensors'
    if target.exists():
        raise FileExistsError(target)
    original = (
        args.source / 'assets/physical-intelligence/libero/norm_stats.json')
    stats = json.loads(original.read_text())['norm_stats']
    (args.output / 'norm_stats.json').write_text(original.read_text())
    (args.output / 'dataset_statistics.json').write_text(
        json.dumps(
            {
                'physical-intelligence/libero': {
                    'proprio': stats['state'],
                    'action': stats['actions']
                }
            },
            indent=2))
    tokenizer = args.output / 'tokenizer'
    tokenizer.mkdir(exist_ok=True)
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/big_vision/paligemma_tokenizer.model',
        tokenizer / 'tokenizer.model')

    print('Restoring official Orbax arrays on CPU', flush=True)
    with ocp.PyTreeCheckpointer() as ckptr:
        path = (args.source / 'params').resolve()
        item = {'params': ckptr.metadata(path)['params']}
        params = ckptr.restore(
            path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(
                        restore_type=np.ndarray, dtype=np.float32),
                    item)))['params']
    flat = flatten(params)
    if all(key[-1] == 'value' for key in flat):
        flat = {key[:-1]: value for key, value in flat.items()}
    paligemma = {
        '/'.join(key[1:]): value
        for key, value in flat.items() if key[0] == 'PaliGemma'
    }
    config = SimpleNamespace(
        vision_config=SimpleNamespace(
            hidden_size=1152,
            num_hidden_layers=27,
            num_attention_heads=16,
            intermediate_size=4304,
            patch_size=14,
            projection_dim=2048),
        text_config=SimpleNamespace(
            hidden_size=2048,
            num_hidden_layers=18,
            num_attention_heads=8,
            head_dim=256,
            intermediate_size=16384))
    backbone, expert = namespace['slice_paligemma_state_dict'](paligemma,
                                                               config)
    expert = namespace['slice_gemma_state_dict'](
        expert,
        SimpleNamespace(width=1024, depth=18, num_heads=8, head_dim=256),
        num_expert=1,
        checkpoint_dir='pi05_libero',
        pi05=True)
    tensors = {**backbone, **expert}
    for key in ('action_in_proj', 'action_out_proj', 'time_mlp_in',
                'time_mlp_out'):
        tensors[f'{key}.weight'] = torch.from_numpy(flat[(key, 'kernel')]).T
        tensors[f'{key}.bias'] = torch.from_numpy(flat[(key, 'bias')])
    # The upstream mapping emits the input embedding only; the tied output
    # lm_head is unused by FluxVLA's policy and need not be duplicated.
    tensors = {key: value.contiguous() for key, value in tensors.items()}
    print(f'Saving {len(tensors)} tensors in float32', flush=True)
    temporary = target.with_suffix('.safetensors.partial')
    save_file(tensors, str(temporary))
    temporary.rename(target)
    (args.output / 'conversion.json').write_text(
        json.dumps(
            {
                'source': 'gs://openpi-assets/checkpoints/pi05_libero',
                'openpi_commit': commit,
                'converter_sha256': hashlib.sha256(
                    source.encode()).hexdigest(),
                'precision': 'float32',
                'tensor_count': len(tensors),
                'action_horizon': 10,
                'discrete_state_input': False,
            },
            indent=2))
    print(f'Converted official checkpoint: {target}', flush=True)


if __name__ == '__main__':
    main()
