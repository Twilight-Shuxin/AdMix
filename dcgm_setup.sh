sudo systemctl --now disable nvidia-dcgm
sudo systemctl --now enable nvidia-dcgm
dcgmi group -c device0
dcgmi group -g 2 -a 0
dcgmi group -c device1
dcgmi group -g 3 -a 1

# dcgmi group -l