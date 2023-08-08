#!/bin/bash

for i in {1..10}
do
	python drqn.py --run_name granasoftmax$i --experiment beron2022_trial -k 10 --episodes 50
done
