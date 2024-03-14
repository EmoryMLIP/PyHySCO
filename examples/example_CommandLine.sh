#!bin/sh

# change the file paths given for --file_1 & --file_2 and --output_dir to match your data location and desired corrected image location
# (python pyhysco.py --help) to see all options
python pyhysco.py --file_1 ../data/156334_v.nii.gz --file_2 ../data/156334_-v.nii.gz --ped 1 --output_dir ../results/