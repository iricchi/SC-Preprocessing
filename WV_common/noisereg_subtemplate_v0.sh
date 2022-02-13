#!/bin/bash
cd $DIREC

for i in "/home/iricchi/Local/Tutorials/Pipeline/template_noiseregression.fsf"; do
    # Include outliers as regressors if needed

    if [ -f $DIREC"/outliers.txt" ]; then          

        sed -e 's@PNMPATH@'$DIREC"/regressors_evlist.txt"'@g' \
            -e 's@OUTDIR@'$DIREC"/noise_regression"'@g' \
            -e 's@DATAPATH@'$DIREC"/mfmri.nii.gz"'@g' \
            -e 's@FILT@'"0"'@g' \
            -e 's@OUTLYN@'"1"'@g' \
            -e 's@NPTS@'"$(fslnvols $DIREC"/mfmri.nii.gz")"'@g' \
            -e 's@OUTLPATH@'$DIREC"/outliers.txt"'@g'  <$i> design_noiseregression.fsf
    else

        sed -e 's@PNMPATH@'$DIREC"/regressors_evlist.txt"'@g' \
            -e 's@OUTDIR@'$DIREC"/noise_regression"'@g' \
            -e 's@DATAPATH@'$DIREC"/mfmri.nii.gz"'@g' \
            -e 's@FILT@'"0"'@g' \
            -e 's@OUTLYN@'"0"'@g' \
            -e 's@NPTS@'"$(fslnvols $DIREC"/mfmri.nii.gz")"'@g'  <$i> design_noiseregression.fsf
    fi
done
