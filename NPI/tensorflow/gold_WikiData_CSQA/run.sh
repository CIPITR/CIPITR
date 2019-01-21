jbsub -cores 1+1 -require k80 -err err/$2_$1.txt -out out/$2_$1.txt -mem 250g -q x86_24h /u/ansarigh/miniconda2/bin/python train.py parameters/parameters_$1.json $2 
