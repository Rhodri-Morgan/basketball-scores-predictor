#!/bin/bash

echo "Ensure using python version and dependencies in README"
echo "THIS SCRIPT WILL DELETE ALL model_data AND extracted_raw_data"
read -r -s -p $'Press enter to continue...\n'

rm -rf extracted_raw_data
mkdir extracted_raw_data
python extract_raw_data.py

rm -rf model_data
mkdir model_data

# You can modify this to test new lagged data
lag=(1 2 3 4 5 6 7 8 9 10)
for l in "${lag[@]}"; do
  python build_team_data.py "$l"
done
