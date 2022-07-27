#!/bin/bash

tput setaf 2; echo "Smoothing started in " $DIREC
tput sgr0;

cd $DIREC
cd Processing

mkdir -p Smoothing
cd Smoothing
cp ../../mfmri_denoised_n.nii.gz .

sct_maths -i mfmri_denoised_n.nii.gz -o sn_mfmri_denoised_2x2x6.nii.gz -smooth 0.85,0.85,2.55
mv sn_mfmri_denoised_2x2x6.nii.gz ../../sn_mfmri_denoised_2x2x6.nii.gz
cd ../..
# Copy header info
fslcpgeom mfmri_denoised.nii.gz sn_mfmri_denoised_2x2x6.nii.gz

tput setaf 2; echo "Done!"