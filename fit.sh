#!/bin/bash

for i in {1..10}
do
	python drqn.py --run_name granb$i --experiment beron2022_trial -k 3 --episodes 50
done
