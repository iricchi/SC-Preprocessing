#!/bin/bash

# Script to run FEAT in all folders (1st level)

cd $DIREC

rm "design_1st_Run"$nrun"_mvpa_flobs.fsf"
                
tput setaf 2; echo "First level FEAT analysis started for Run " $nrun 
tput sgr0;  
        
# Generate fsf file from template
for i in $FSL_TEMP"/template.fsf"; do

    if [ -f $DIREC"/outliers.txt" ]; then # If outliers file exists
            sed -e 's@PNMPATH@'$DIR_PHYSIO"/Run"$nrun"_evlist.txt"'@g' \
                -e 's@THRESHPATH@'$DIR_FUNC"/Segmentation/mask_sc.nii.gz"'@g' \
                -e 's@OUTDIR@'"output_feat_first_flobs"'@g' \
                -e 's@DATAPATH@'"/srmfmri_2x2x6.nii.gz"'@g' \
                -e 's@NPTS@'"$(fslnvols srmfmri_2x2x6.nii.gz)"'@g' \
                -e 's@OUTLYN@'"1"'@g' \
                -e 's@ICAYN@'"0"'@g' \
                -e 's@OUTLPATH@'$DIREC"/nuisance.txt"'@g' \
                -e 's@TIMEPATH_L@'$PAR_DIR"/timing/left_"$nrun".txt"'@g' \
                -e 's@TIMEPATH_R@'$PAR_DIR"/timing/right_"$nrun".txt"'@g' <$i> "design_1st_Run"$nrun"_mvpa_flobs.fsf"
    else
            sed -e 's@PNMPATH@'$DIR_PHYSIO"/Run"$nrun"_evlist.txt"'@g' \
                -e 's@THRESHPATH@'$DIR_FUNC"/Segmentation/mask_sc.nii.gz"'@g' \
                -e 's@OUTDIR@'"output_feat_first_flobs"'@g' \
                -e 's@DATAPATH@'$DIREC"/srmfmri_2x2x6.nii.gz"'@g' \
                -e 's@NPTS@'"$(fslnvols srmfmri_2x2x6.nii.gz)"'@g' \
                -e 's@OUTLYN@'"0"'@g' \
                -e 's@ICAYN@'"0"'@g' \
                -e 's@OUTLPATH@'""'@g' \
                -e 's@TIMEPATH_L@'$PAR_DIR"/timing/left_"$nrun".txt"'@g' \
                -e 's@TIMEPATH_R@'$PAR_DIR"/timing/right_"$nrun".txt"'@g' <$i> "design_1st_Run"$nrun"_mvpa_flobs.fsf"

    fi
done

# Run the analysis using the newly created fsf file
fsl5.0-feat "design_1st_Run"$nrun"_mvpa_flobs.fsf"
