import subprocess

def start_dcgm():
    cmd = f'sudo systemctl --now enable nvidia-dcgm'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()

def stop_dcgm():
    cmd = f'sudo systemctl --now disable nvidia-dcgm'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()

def add_dcgm_group(group_name):
    cmd = f'dcgmi group -c {group_name}'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = p.communicate()
    group_id = int(stdout.strip().split()[-1])
    return group_id

def add_devices_to_dcgm_group(dcgm_group_id, device_ids):
    cmd = f'dcgmi group -g {dcgm_group_id} -a {','.join(device_ids)}'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    p.wait()

def delete_dcgm_group(dcgm_group_id):
    cmd = f'dcgmi group -d {dcgm_group_id}'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    p.wait()

def list_dcgm_group():
    cmd = f'dcgmi group -l'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = p.communicate()
    print(stdout.strip())