#!/bin/bash

cd $DIREC

mkdir TA
cd TA

cp ../s_mfmri_denoised_2x2x6.nii.gz .
cp ../Segmentation/mask_sc.nii.gz .
gunzip mask_sc.nii.gz

# First, split data
fslsplit s_mfmri_denoised_2x2x6.nii.gz resvol -t
tput setaf 2; echo "...Unzip"
		tput sgr0;

for n in "$PWD"/resvol*.nii.gz; do # Loop through all files
        	IFS='.' read -r volname string <<< "$n"
        	gunzip "${volname##*/}".nii.gz
	done

tput setaf 2; echo "...Clean"
   	 	tput sgr0;

rm resvol*.nii.gz