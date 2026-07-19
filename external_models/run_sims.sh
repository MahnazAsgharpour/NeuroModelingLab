#!/bin/bash

Emin=60
Emax=300
Estep=20
N=20
configdir="configs"
outdir="sims"

for i in `seq $N` ; do
    for mag in `seq $Emin $Estep $Emax` ; do
	config=$configdir/Efield_${i}_${mag}.json
	outfile=$outdir/Efield_${i}_${mag}.npz
	python3 simulate_cell.py --save-voltage-traces -o $outfile $config &
    done
    wait
done
