#!/usr/bin/bash

# sbatch -N 1 -o SFI_TESTS -t 06:00:00 -p RM mon.sh
../../humanities-glass/MPF_CMU/mpf -c monitoring.dat 1.0
