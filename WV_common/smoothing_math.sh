#!/bin/bash

tput setaf 2; echo "Smoothing started in " $DIREC
tput sgr0;

cd $DIREC
cd Processing

mkdir -p Smoothing
cd Smoothing
cp ../../"$MFMRID"_n.nii.gz .

# split on the time axis 
fslsplit "$MFMRID"_n.nii.gz denoised -t
files_to_smooth="$PWD"/denoised*.nii.gz

j=0
for den in $files_to_smooth; do # Loop through all files
    sct_maths -i $den -o "tmp_sn_"$MFMRID"_"$j".nii.gz" -smooth 0.85,0.85,2.55
    ((j=j+1))
    rm $den
done

# merge back in time
fslmerge -t sn_"$MFMRID"_2x2x6.nii.gz tmp_sn_"$MFMRID"_*.nii.gz
rm tmp_sn_"$MFMRID"_*.nii.gz

mv sn_"$MFMRID"_2x2x6.nii.gz ../../sn_"$MFMRID"_2x2x6.nii.gz
cd ../..
# Copy header info
fslcpgeom $MFMRID.nii.gz sn_"$MFMRID"_2x2x6.nii.gz

tput setaf 2; echo "Done!"