jbsub  -err err/$2_$1.txt -out out/$2_$1.txt -cores 1+1 -require k80 -mem 100g -q x86_1h /dccstor/cssblr/amrita/miniconda/bin/python -W ignore train.py parameters/parameters_$1.json $2 
