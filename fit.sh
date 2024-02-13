#!/bin/bash

for i in {1..10}
do
	echo '========================'
	echo $i
	echo '========================'
	python drqn.py --run_name lowgamma_$i --experiment beron2022_trial -k 10 -g 0.2
done
