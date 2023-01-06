#!/bin/bash

tput setaf 2; echo "Smoothing started in " $DIREC
tput sgr0;

cd $DIREC
cd Processing

mkdir -p Smoothing
cd Smoothing
cp ../../mfmri_denoised_n.nii.gz .

# split on the time axis 
fslsplit mfmri_denoised_n.nii.gz denoised -t
files_to_smooth="$PWD"/denoised*.nii.gz

i=0
for den in $files_to_smooth; do # Loop through all files
    sct_maths -i $den -o "sn_mfmri_denoised_"$i".nii.gz" -smooth 0.85,0.85,2.55
    rm $den
done

# merge back in time
fslmerge -t sn_mfmri_denoised_2x2x6.nii.gz sn_mfmri_denoised_*.nii.gz
rm sn_mfmri_denoised_*.nii.gz

# sct_maths -i mfmri_denoised_n.nii.gz -o sn_mfmri_denoised_2x2x6.nii.gz -smooth 0.85,0.85,2.55

mv sn_mfmri_denoised_2x2x6.nii.gz ../../sn_mfmri_denoised_2x2x6.nii.gz
cd ../..
# Copy header info
fslcpgeom mfmri_denoised.nii.gz sn_mfmri_denoised_2x2x6.nii.gz

tput setaf 2; echo "Done!"