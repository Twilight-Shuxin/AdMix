#!/bin/bash

# usage: group to monitor is currently hard-coded to be group 3; give the log_file, the item to monitor

nohup dcgmi dmon -g 3 -e $1 > $2 &
echo $! > tmp.txt