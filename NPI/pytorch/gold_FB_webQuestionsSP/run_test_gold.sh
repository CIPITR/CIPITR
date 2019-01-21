jbsub -cores 1+1 -require k80 -err err/$2_$1_test_gold.txt -out out/$2_$1_test_gold.txt -mem 60g -q x86_24h /dccstor/cssblr/amrita/miniconda/bin/python load.py parameters/gold/parameters_$1.json $2
