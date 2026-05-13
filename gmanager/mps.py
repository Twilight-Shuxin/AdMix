import os
import subprocess
from utils import *

def get_hash(UUID_list):
    n = len(UUID_list)
    length = len(UUID_list[0])
    part_size = length // n
    result = ""
    for i, s in enumerate(UUID_list):
        start = i * part_size
        # Last string takes remainder to preserve full length
        if i == n - 1:
            end = length
        else:
            end = (i + 1) * part_size
        result += s[start:end]
    return result

def prefix_mps(UUID_list):
    if(not UUID_list):
        return f''
    UUID_hash = get_hash(UUID_list)
    return f'export CUDA_VISIBLE_DEVICES={','.join(UUID_list)} && export  CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-{UUID_hash} && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-{UUID_hash} && '

def open_mps(UUID_list = []):
    cmd = prefix_mps(UUID_list) + f'nvidia-cuda-mps-control -d'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    p.wait()
    read = str(p.stdout.read())

    gmanager_home = os.environ["GM_HOME"]
    if(gmanager_home.endswith("/")):
        gmanager_home = gmanager_home[:-1]
    cmd = prefix_mps(UUID_list) + f'python {gmanager_home}/warmup.py'
    print(prefix_mps(UUID_list))
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    p.wait()
    print(str(p.stdout.read()))


def close_mps(UUID_list = []):
    cmd = prefix_mps(UUID_list) + f'echo quit | nvidia-cuda-mps-control'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    p.wait()
    read = str(p.stdout.read())


def set_mps_percentage(UUID_list, Percentage):
    cmd = prefix_mps(UUID_list) + f'echo get_server_list | nvidia-cuda-mps-control'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    p.wait()
    read = str(p.stdout.read().decode())
    server_ID = int(read)

    cmd = prefix_mps(UUID_list) + f'echo set_active_thread_percentage {server_ID} {Percentage} | nvidia-cuda-mps-control'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    p.wait()

def get_mps_percentage(UUID_list = []):
    cmd = prefix_mps(UUID_list) + f'echo get_server_list | nvidia-cuda-mps-control'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    p.wait()
    read = str(p.stdout.read().decode())
    server_ID = int(read)

    cmd = prefix_mps(UUID_list) + f'echo get_active_thread_percentage {server_ID} | nvidia-cuda-mps-control'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    p.wait()
    read = str(p.stdout.read().decode().strip())
    SM_percentage = float(read)
    return SM_percentage

def check_mps_started(UUID_list = []):
    cmd = prefix_mps(UUID_list) + f'echo get_server_list | nvidia-cuda-mps-control'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = p.communicate()
    if stdout:
        return True
    elif stderr:
        return False