#!/bin/bash

for i in {1..10}
do
	python drqn.py --run_name granz$i --experiment beron2022_time -k 10 --episodes 50
done
