#!/bin/bash

for i in {1..3}
do
	sbatch $i.sh
	echo "Job $i submitted"
done