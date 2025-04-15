#!/bin/bash

# usage: group to monitor is currently hard-coded to be group 2; give the log_file, the item to monitor

nohup dcgmi dmon -g $1 -e $2 > $3 &
echo $! > tmp.txt