jbsub -cores 1+1 -require k80 -err err/$2_$1_test.txt -out out/$2_$1_test.txt -mem 250g -q x86_24h /u/ansarigh/miniconda2/bin/python load.py parameters/parameters_$1.json $2 
