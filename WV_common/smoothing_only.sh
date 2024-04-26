#!/bin/bash

DIREC="/media/miplab-nas2/Data/SpinalCord/3_RestingState/LongRecordings/Cervical/iCAPs_Ila" 

tput setaf 2; echo "Smoothing started in " $DIREC
tput sgr0;

cd $DIREC

mkdir -p Smoothing
cd Smoothing


files="$DIREC"/RS_*.nii.gz
i=0
for f in $files; do
    mkdir -p sub_"$i"
    cd sub_"$i"
    cp $f .
    fslsplit RS_*.nii.gz denoised -t
    files_to_smooth="$PWD"/denoised*.nii.gz
    
    j=0
    for den in $files_to_smooth; do
        sct_maths -i $den -o sn_mfmri_denoised_"$j".nii.gz -smooth 0.85,0.85,2.55
        rm $den
        ((j=j+1))
    done

    fslmerge -t sn_mfmri_2x2x6_"$i".nii.gz sn_mfmri_denoised*.nii.gz 
    rm sn_mfmri_denoised*.nii.gz

    
    mv sn_mfmri_2x2x6_"$i".nii.gz ../sn_mfmri_2x2x6_"$i".nii.gz
    cd ..
    fslcpgeom $f sn_mfmri_2x2x6_"$i".nii.gz
    ((i=i+1))
done


tput setaf 2; echo "Done!"