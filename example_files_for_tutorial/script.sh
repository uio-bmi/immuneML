#!/bin/bash

echo "Generating repertoires..."

mkdir repertoires

echo filename,identifier,subject_id > metadata.csv

for i in {1..100}
do
   olga-generate_sequences --humanTRB -n 1000 -o repertoires/rep$i.tsv
   echo repertoires/rep$i.tsv,rep$i,rep$i >> metadata.csv
done

echo "Finished generating repertoires"