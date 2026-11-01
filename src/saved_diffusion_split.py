"""Read held-out membership from a checkpoint's saved split, without resplitting."""
import hashlib
import json
from pathlib import Path


def _name(value):
    name = Path(str(value)).name
    for suffix in ('.label.nii.gz', '.label.nii'):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def select_saved_test_paths(label_paths, split_json, subset_size=50, subset_offset=0):
    if not split_json:
        raise ValueError('--diffusion_split_json is required for a held-out subset; use the checkpoint training split')
    path = Path(split_json)
    raw = path.read_bytes()
    split = json.loads(raw)
    names = {}
    for partition in ('train', 'val', 'test'):
        values = split.get(partition, split.get(partition + '_paths'))
        if not isinstance(values, list) or not values:
            raise ValueError(f'Saved split must contain a nonempty {partition} list')
        names[partition] = [_name(v) for v in values]
        if len(set(names[partition])) != len(values):
            raise ValueError(f'Duplicate cases in saved {partition} split')
    if set(names['test']) & (set(names['train']) | set(names['val'])):
        raise ValueError('Saved test cases overlap training or validation cases')
    if set(names['train']) & set(names['val']):
        raise ValueError('Saved training and validation cases overlap')
    size, offset = int(subset_size), int(subset_offset)
    if size <= 0 or offset < 0 or offset + size > len(names['test']):
        raise ValueError('Requested subset lies outside the saved test split')
    available = {}
    for label in label_paths:
        name = _name(label)
        if name in available:
            raise ValueError(f'Duplicate local mask identity: {name}')
        available[name] = label
    selected = names['test'][offset:offset + size]
    missing = [name for name in selected if name not in available]
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} requested test masks; first missing: {missing[0]}')
    return [available[name] for name in selected], {
        'selection_rule': 'contiguous entries in the saved test-list order; no new random split',
        'saved_split_json': str(path),
        'saved_split_sha256': hashlib.sha256(raw).hexdigest(),
        'n_train': len(names['train']), 'n_val': len(names['val']), 'n_test': len(names['test']),
        'subset_size': size, 'subset_offset': offset, 'subset_end_exclusive': offset + size,
        'source_anatomy_training_independence_claimed': False,
    }
