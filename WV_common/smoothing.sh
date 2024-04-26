#!/bin/bash

tput setaf 2; echo "Smoothing started in " $DIREC
tput sgr0;

cd $DIREC
cd Processing

mkdir -p Smoothing
cd Smoothing

# First, isolate slice init 
# if [ ! -f slice_init.nii.gz ]; then
/usr/local/fsl/bin/fslroi ../../$MFMRID slice_init 0 -1 0 -1 0 1 0 -1
#fi
dim=$(/usr/local/fsl/bin/fslsize ../../$MFMRID)
arr=(`echo ${dim}`)
z=${arr[5]}
maxz=$(( z-1 ))

# Then, isolate slice last
#if [ ! -f slice_last.nii.gz ]; then
/usr/local/fsl/bin/fslroi ../../$MFMRID slice_last 0 -1 0 -1 $maxz 1
#fi


# And for mask as well
#if [ ! -f ../../Segmentation/mask_init.nii.gz ]; then
/usr/local/fsl/bin/fslroi ../../Segmentation/$MASK_NAME ../../Segmentation/mask_init 0 -1 0 -1 0 1 0 -1
#fi

#if [ ! -f ../../Segmentation/mask_last.nii.gz ]; then
/usr/local/fsl/bin/fslroi ../../Segmentation/$MASK_NAME ../../Segmentation/mask_last 0 -1 0 -1 $maxz 1
#fi

#rm residuals_native_padded.nii.gz

#if [ ! -f "$MFMRID"_padded.nii.gz ]; then
/usr/local/fsl/bin/fslmerge -z "$MFMRID"_padded ../../$MFMRID slice_last slice_last slice_last slice_last slice_last
/usr/local/fsl/bin/fslmerge -z "$MFMRID"_padded slice_init slice_init slice_init slice_init slice_init "$MFMRID"_padded
#fi

#if [ ! -f mask_padded.nii.gz ]; then
/usr/local/fsl/bin/fslmerge -z mask_padded ../../Segmentation/$MASK_NAME ../../Segmentation/mask_last ../../Segmentation/mask_last ../../Segmentation/mask_last ../../Segmentation/mask_last ../../Segmentation/mask_last
/usr/local/fsl/bin/fslmerge -z mask_padded ../../Segmentation/mask_init ../../Segmentation/mask_init ../../Segmentation/mask_init ../../Segmentation/mask_init ../../Segmentation/mask_init mask_padded
#fi

# Create temporary folder
if [ ! -d "./tmp" ]; then
        mkdir -p tmp
else 
        rm -rf tmp
fi

# Copy input data to tmp folder & split
cp "$MFMRID"_padded.nii.gz tmp
cd tmp
sct_image -i "$MFMRID"_padded.nii.gz -split t -o "$MFMRID"_padded.nii.gz
rm "$MFMRID"_padded.nii.gz

files_to_smooth="$PWD"/*
for b in $files_to_smooth; do # Loop through all files
        sct_smooth_spinalcord -i $b -s ../mask_padded.nii.gz -smooth 0.85,0.85,2.55
        rm $b
done

# Concat, put back in initial folder and delete temporary folder
/usr/local/fsl/bin/fslmerge -t s_"$MFMRID"_padded_2x2x6 "$MFMRID"_padded*_smooth.*
mv s_"$MFMRID"_padded_2x2x6.nii.gz ..
cd ..

sct_crop_image -i s_"$MFMRID"_padded_2x2x6.nii.gz -zmin 5 -zmax -6 -o s_"$MFMRID"_2x2x6.nii.gz
mv s_"$MFMRID"_2x2x6.nii.gz ../../s_"$MFMRID"_2x2x6.nii.gz

cd ../..

# Copy header info
/usr/local/fsl/bin/fslcpgeom "$MFMRID".nii.gz s_"$MFMRID"_2x2x6.nii.gz

tput setaf 2; echo "Done!"