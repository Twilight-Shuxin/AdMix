from collections import defaultdict

def avg_dict(dicts):
    """
        dicts: list of dicts
        return: dict with averaged values
    """
    sums = defaultdict(float)
    counts = defaultdict(int)
    for data in dicts:
        for k, v in data.items():
            sums[k] += v
            counts[k] += 1
    return {k: sums[k] / counts[k] for k in sums}