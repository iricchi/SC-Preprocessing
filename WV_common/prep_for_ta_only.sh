#!/bin/bash
DIREC="/media/miplab-nas2/Data/SpinalCord/3_RestingState/LongRecordings/Cervical/iCAPs_Ila" 

tput setaf 2; echo "TA preparation started in " $DIREC
tput sgr0;

cd $DIREC

subjects="$DIREC"/Smoothing/sub_*

i=0
for sub in $subjects; do
    cd $sub 
    mkdir TA
    cd TA
    cp ../../sn_mfmri_2x2x6_"$i".nii.gz .
    cp ../../../mask/PAM50_cord_crop_z.nii.gz .
    gunzip PAM50_cord_crop_z.nii.gz

    # first split data
    fslsplit sn_mfmri_2x2x6_"$i".nii.gz resvol -t
    tput setaf 2; echo "...Unzip"
		tput sgr0;

    for n in "$PWD"/resvol*.nii.gz; do # LoThop through all files
        IFS='.' read -r volname string <<< "$n"
        gunzip "${volname##*/}".nii.gz
	done
    tput setaf 2; echo "...Clean"
   	 	tput sgr0;

    rm resvol*.nii.gz

    ((i=i+1))
done 
