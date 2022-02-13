#!/bin/bash
#
# Script to smooth realigned fMRI images (rmfmri.nii.gz) 
# Smoothing is done along the spinal cord with a 2x2x6mm FWHM Gaussian kernel

cd $DIREC

tput setaf 2; echo "Smoothing started in " $run 
tput sgr0;

cd Processing

mkdir -p Smoothing
cd Smoothing

# We need to know the dimension
dim=$(fslsize ../../rmfmri.nii.gz)
arrdim=($dim)
z=${arrdim[5]}
maxz=$(( z-1 ))

# First, isolate slice last
if [ ! -f slice_last.nii.gz ]; then
        tput setaf 2; echo "Image dimension (z) " $maxz

        tput sgr0;
        fslroi ../../rmfmri.nii.gz slice_last 0 -1 0 -1 $maxz 1
fi


 # Then, isolate slice init 
if [ ! -f slice_init.nii.gz ]; then
        fslroi ../../rmfmri slice_init 0 -1 0 -1 0 1 0 -1
fi

if [ ! -f ../../../Segmentation/mask_last.nii.gz ]; then
        fslroi ../../../Segmentation/mask_sc.nii.gz ../../../Segmentation/mask_last 0 -1 0 -1 $maxz 1
fi

# And for mask as well
if [ ! -f ../../../Segmentation/mask_init.nii.gz ]; then
        fslroi ../../../Segmentation/mask_sc.nii.gz ../../../Segmentation/mask_init 0 -1 0 -1 0 1 0 -1
fi

if [ ! -f rmfmri_padded.nii.gz ]; then
        fslmerge -z rmfmri_padded ../../rmfmri slice_last slice_last slice_last slice_last slice_last
        fslmerge -z rmfmri_padded slice_init slice_init slice_init slice_init slice_init rmfmri_padded
fi

if [ ! -f mask_padded.nii.gz ]; then
        fslmerge -z mask_padded ../../../Segmentation/mask_sc ../../../Segmentation/mask_last ../../../Segmentation/mask_last ../../../Segmentation/mask_last ../../../Segmentation/mask_last ../../../Segmentation/mask_last
        fslmerge -z mask_padded ../../../Segmentation/mask_init ../../../Segmentation/mask_init ../../../Segmentation/mask_init ../../../Segmentation/mask_init ../../../Segmentation/mask_init mask_padded
fi

# Create temporary folder
rm -r tmp
mkdir -p tmp

# Copy input data to tmp folder & split
cp rmfmri_padded.nii.gz tmp
cd tmp
sct_image -i rmfmri_padded.nii.gz -split t -o rmfmri_padded.nii.gz
rm rmfmri_padded.nii.gz

files_to_smooth="$PWD"/*
for b in $files_to_smooth; do # Loop through all files
        sct_smooth_spinalcord -i $b -s ../mask_padded.nii.gz -smooth 0.85,0.85,2.55
        rm $b
done

# Concat, put back in initial folder and delete temporary folder
fslmerge -t srmfmri_padded_2x2x6 rmfmri_padded*_smooth.*
mv srmfmri_padded_2x2x6.nii.gz ..
cd ..
#rm -r tmp

sct_crop_image -i srmfmri_padded_2x2x6.nii.gz -zmin 5 -zmax -6  -o srmfmri_2x2x6.nii.gz
rm srmfmri_padded_2x2x6.nii.gz
mv srmfmri_2x2x6.nii.gz ../../srmfmri_2x2x6.nii.gz

cd ../..
# Copy header info
fslcpgeom rmfmri.nii.gz srmfmri_2x2x6.nii.gz


