import re
import time
import subprocess

def get_gpu_uuid(gpu_id):
    process = subprocess.Popen(['nvidia-smi', '-L'], stdout=subprocess.PIPE, text=True)
    flag = False
    end_flag = False
    UUID_list = []
    while True:
        line = process.stdout.readline()
        if not line:
            break
        gpu_pattern = re.compile(rf'GPU {gpu_id}: .* \(UUID: GPU-(.*?)\)')
        gpu_match = gpu_pattern.search(line)
        if gpu_match:
            match = re.search(r'GPU-[\da-fA-F\-]+', line.strip())
            uuid = match.group()
            return uuid
    return None

def get_gpu_uuids(gpu_ids):
    return [get_gpu_uuid(gpu_id) for gpu_id in gpu_ids]