from collections import defaultdict

def avg_dict(dicts):
    """
        dicts: list of dicts
        return: dict with averaged values (handles nested dicts recursively)
    """
    sums = defaultdict(float)
    counts = defaultdict(int)
    nested = defaultdict(list)

    for data in dicts:
        for k, v in data.items():
            if isinstance(v, dict):
                nested[k].append(v)
            elif isinstance(v, (int, float)):
                sums[k] += v
                counts[k] += 1

    result = {k: sums[k] / counts[k] for k in sums}
    for k, v in nested.items():
        result[k] = avg_dict(v)
    return result