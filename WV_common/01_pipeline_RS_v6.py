# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
IMPORTANT NOTE: The denoising has been replaced from FSL feat with nilearn for better compatiblity and
for fixing issues like temporal filtering on regressors.

This pipeline is used mainly for RESTING STATE.

Order of the options (recommended to follow):

STEPS :

    // = could be done in parallel since one is one anat and the other on func
 
    0) Segmentation with SCT and labels generation (outside the pipeline)
    Commands used:
        sct_deepseg_sc with viewer initialization
        sct_deepseg_sc -i t2.nii.gz -c t2 -centerline viewer

        ### Lumbar
        sct_deepseg -task seg_lumbar_sc_t2w -i t2.nii.gz 
        sct_label_utils -i t2.nii.gz -create 32,257,374,99 / 60 substitued 99  -o label_caudaequinea.nii.gz
        ###

        sct_label_utils -i t2.nii.gz -create-viewer 2,3,4,5,6,7,8,9,10,11,12 -o labels.nii.gz
        or this to generate labels for normalization 

        sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -c t2


        But here I have always labelled the spinal levels so the option of the anatomical 
        normalization and registration to template is '-lspinal'.

        For Lumbar (if you add CSF):
        sct_propseg -i t2.nii.gz -c t2 -o t2_seg.nii.gz -CSF
        -
        fslmaths t2_seg.nii.gz -add t2_rescaled_CSF_seg.nii.gz t2_seg_CSF.nii.gz
        sct_label_utils -i t2.nii.gz -create-viewer 18,19,20,21,22,23,24,25 -o labels.nii.gz  
    
    Preprocessing Pipeline:
    1) anat_norm (register to template, anatomical normalization) //
    2) moco (motor correction) // run also the motor correction jupyter notebook to see who to remove 
    3) func_norm (functional images normalization)
    4) pnm (physiological noise modelling)
    5) denoising (remove noise due to movements...)
        NOTE (README): added export path for python version because of fsl feat 5 that created 
        problems in the specific  
    6) normalization for population study especially (but could be run also subject-level)
        [This step initially was run after iCAPs, in case the normalization should be run
        after iCAPs, do not use this script but the external scripts in bash and matlab: 
        'a_normalization_after_ta.sh' and 'b_normalization_after_ta.m']
   
     You can also crop PAM50 to have a template that is not the whole spine otherwise too long
    (this could be in the local directory for example on the Lumbar project on the server 
    ~/PhDProjects/Lumbar/templates)
    run:
     sct_crop_image -i PATH_TO_PAM50_t2 -o PAM50_t2_cropped.nii.gz -zmin 0 -zmax XX
     sct_warp_template -d PAM50_t2_cropped.nii.gz -w transform_identity.txt -s 1 -ofolder PATH_OUTPUT

    7) smoothing (with sct_math if normalization has been run or sct_smooth_spinal_cord otherwise (for subject-wise analysis) ) 
    This step is particularly important for the TA and iCAPs

    8) prep_for_ta (prepare niftii images for iCAPs)
    
    On matlab...
    9) iCAPs 
        - run Total Activation
        - run Thresholding
        - run Clustering (per subject specifying one by one)

       then for whole subjects-wise analysis (run 03_Post_iCAPs)
        - sct_apply_transfo -w ../../../../sub-001/func/Normalization/warp_fmri2anat.nii.gz -i iCAP_z_1.nii -d t2.nii.gz
         and on iCAPs_z total
        

    10) run Clustering on all subjects list
    11) run post_iCAPs for Native so that you have 

Usage:
    python pipeline.py --config config_file.json 
    or
    python pipeline.py --config config_file.json --anat_norm --moco --func_norm
    (to run specific processes)


WARNING1: if you don't specify the options --anat_norm, --moco, the flags used will be the ones
specified in the config file.
The options (eg --moco) are used to set some specific processes to true, otherwise you can
declare them on the config file.

WARNING2: it's important to open the config file in order to set all the other variables properly.
The options --operation are used only for the preprocessing.

NOTE: The flags should NOT be set all to true at the same time because some processes need to be
checked manually (e.g. pnm)


Example:
    python pipeline.py --config config_preprocess.json --moco (will set motor correction to true)

"""

import os
import sys
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
from glob import glob
from joblib import Parallel, delayed
import subprocess
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from nipype.algorithms.confounds import CompCor
import nibabel as nib
from nibabel import load, save, Nifti1Image
import nilearn as nil 
from nilearn.image import clean_img
from nilearn.signal import clean
from scipy.signal import butter, filtfilt, cheby2, lfilter
from sklearn import preprocessing as pr
import requests

global YFACTOR, TRG

TRG = 1 # 5 IN CB, 1 FOR KAPTAN'S
YFACTOR = 100  # 10 for CB and Rob's , 100 for Kaptan's

def _send_message_telegram(whatever):

    TOKEN = "5979998311:AAGl9Did2fwB_1rEd0RHA496Qox6xLghrOM"
    chat_id = "626146439"
    message = f"Pipeline: I'm done with {whatever}! "
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    print(requests.get(url).json()) # this sends the message


class PreprocessingRS(object):
    """PreprocessingRS performs the STEP1 for the whole Pipeline in RS.

    NOTE: segmentation and labelling are done separately and should be visually evaluated!
          FSLeyes was used to normalize manually the functional images, by creating a mask


    """
    def __init__(self, config_file):
        super(PreprocessingRS, self).__init__()
        self.config_file = config_file
        self.__init_configs()
        self.working_dir = os.getcwd()
        self.__read_configs(config_file)


    def __init_configs(self):
        """Initialze variables"""

        self.version = 0 
        self.TR = 2.5
        self.fmriname = "fmri"  # this is the name of the fmri nifti to analyse
        self.subs2exclude = []

        self.anat_norm = False
        self.slice_timing = False
        self.moco = False
        self.mocopoly=0
        self.func_norm = False
        self.mask_fname = "mask_sc"
        self.mask_csf_name = "mask_csf"
        self.pnm0_preptxt = False   # step 0: prepare physio rec (txt file and filter)
        self.pnm1_stage1 = False   # first step: pnm_stage 1
        self.pnm2_peaks = False   # second step: check peaks of cardiac sig
        self.pnm3_gen_evs = False   # third step: generate evs 
        self.cof = 2  # default filter parameter for pnm0
        self.filter_cardiac = False   # by default don't filter cardiac data
        self.mode = ''
        self.fsessions_name = ''
        self.csf_mask = False  # default / in cervical should be true
        self.filter_evs = '' # by default don't filter, otherwise BP or HP 
        self.denoising = False
        self.denoising_regs = [] # can contain compcor (PCA-like additional physio noise regs)
        self.standardize_ts = False # input for clean_img , should be false if you want to see sig variance
        self.normalization = False
        self.transf_type = "spline"
        self.smoothing = False
        self.s_zsize = 6
        self.smooth_after_norm = False
        self.prep_for_ta = False
        self.pam50_path = "/home/iricchi/sct_5.3.0/data/PAM50/template"
        self.pam50_template = "PAM50_t2"
        self.pam50_maskname = ""

    def processes(self):
        if self.anat_norm:
            self.anat_seg_norm()
            os.chdir(self.working_dir)

        if self.slice_timing:
            self.slice_timing_corr()
            os.chdir(self.working_dir)

        if self.moco:
            self.motor_correction()
            os.chdir(self.working_dir)

        if self.func_norm:
            self.func_normalize()
            os.chdir(self.working_dir)

        if self.pnm0_preptxt:
            self.prepare_physio()
            os.chdir(self.working_dir)

        if self.pnm1_stage1:
            self.pnm_stage1()
            os.chdir(self.working_dir)

        if self.pnm2_peaks:
            self.pnm_stage2()
            os.chdir(self.working_dir)

        if self.pnm3_gen_evs:
            self.generate_evs()
            os.chdir(self.working_dir)

        if self.denoising:
            self.apply_denoising()

        if self.normalization:
            self.normalize()

        if self.smoothing:
            self.apply_smoothing()

        if self.prep_for_ta:
            self.prepare_for_ta()


    def __read_configs(self, input_file):
        """Recursively read configs from given JSON file."""
        with open(input_file) as js_file:
            
            params = json.load(js_file)

        for key in params:
            if hasattr(self, key) and isinstance(getattr(self, key), list):
                # extend if parameter is a list
                assert isinstance(params[key], list)
                # setattr(self, key, getattr(self, key).extend(params[key]))
                getattr(self, key).extend(params[key])
            else:
                # initialize or overwrite valuex
                setattr(self, key, params[key])

        # assign the list of subjects variable
        self.list_subjects = glob(os.path.join(self.parent_path, self.data_root+'*'))
        
        subj_found = [sub.split('/')[-1] for sub in self.list_subjects]
        print("Subjects found:", subj_found)
        
        # remove subjects
        print("Removing subjects:", self.subs2exclude)
        self.list_subjects = [sub for sub in self.list_subjects if sub.split('/')[-1] not in self.subs2exclude]

        
        # # # #### comment
        # selected_subjects = ['sub-02'] #,'sub-09', 'sub-10','sub-11','sub-12', 'sub-13', 'sub-14', 'sub-15'] 
        # selected_subjects = ['sub-ZS001','sub-ZS002','sub-ZS003']
        # self.list_subjects = [sub for sub in self.list_subjects if sub.split('/')[-1]  in selected_subjects]
        # # # # ###  end comment

        print("Tot num of subjects:", len(self.list_subjects))


        ### temporal filtering section
        self.donotfilter = False
        if self.filter_evs == "BP13":
            self.Wn = [0.01, 0.13]
        elif self.filter_evs == "BP17":
            self.Wn = [0.01, 0.17]
        elif self.filter_evs not in ["BP13","BP17","HP"]:
            self.donotfilter = True

    def anat_seg_norm(self):
        subj_paths = [os.path.join(s, self.anat) for s in self.list_subjects]
       
        ## TO COMMENT AFTER
        ###
        # list_subjects = ['LU_AT', 'LU_EP', 'LU_FB', 'LU_GL', 'LU_GP', 'LU_MD', 'LU_VS']
        # subj_paths = [os.path.join(self.parent_path, s, self.anat) for s in list_subjects]
        ###
        
        # ## (un)comment
        # s = self.list_subjects[0]
        # print(s)
        # subj_paths = [os.path.join(s, self.anat)]
        # ##
        
        start = time.time()
        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="multiprocessing")(delayed(self._register_to_template)(sub)\
                 for sub in subj_paths)

        print("### Normalization Done!")
        print("### Info: the normalization took %.3f s" %(time.time()-start))

        return

    def _register_to_template(self, sps):

        # sps = specific path subject
        
        ##### NOTE:
        ## for JC version2 from SCT forum
        ## REMEMBER TO RUN THIS ON TERMINAL BEFORE IN CASE NOT DONE
        ## Add a label of arbitrary value 99 to the PAM50 template at the start of the cauda equinea
        # sct_label_utils -i $SCT_DIR/data/PAM50/template/PAM50_label_disc.nii.gz -create-add 70,69,46,99 -o $SCT_DIR/data/PAM50/template/PAM50_label_disc.nii.gz
        #####

        # No particular normalization used
        #### This was for Lumbar ###
        # run_string0 = f'cd {sps}; sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -ldisc labels.nii.gz -c t2 \
        # -param step=0,type=label,dof=Tx_Ty_Tz_Sz:step=1,type=seg,algo=centermassrot,metric=MeanSquares,iter=10,smooth=2,gradStep=0.5,slicewise=0,smoothWarpXY=2,pca_eigenratio_th=1.6:step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,iter=3,smooth=1,slicewise=0,gradStep=0.5,smoothWarpXY=2,pca_eigenratio_th=1.6:step=3,type=im,metric=CC'
        
        ### vertebrae labels were used for the cervical 

        run_string0 = f'cd {sps}; sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -ldisc labels.nii.gz -c t2 \
         -param step=0,type=label,dof=Tx_Ty_Tz_Sz:step=1,type=seg,algo=centermassrot,metric=MeanSquares,iter=10,smooth=2,gradStep=0.5,slicewise=0,smoothWarpXY=2,pca_eigenratio_th=1.6:step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,iter=3,smooth=1,slicewise=0,gradStep=0.5,smoothWarpXY=2,pca_eigenratio_th=1.6:step=3,type=im,metric=CC'
        
        

        ## no special params
        # run_string = 'cd %s; sct_register_to_template -i t2.nii.gz -s t2_seg_CSF.nii.gz -ldisc labels.nii.gz -c t2 -param step=0,type=label,dof=Tx_Ty_Tz_Sz:step=1,type=seg,algo=centermassrot,metric=MeanSquares,iter=10,smooth=2,gradStep=0.5,slicewise=0,smoothWarpXY=2,pca_eigenratio_th=1.6' % sps

        ## modifying params
        # run_string = 'cd %s; sct_register_to_template -i t2_cropped.nii.gz -s t2_seg_CSF_cropped.nii.gz -ldisc labels.nii.gz -c t2 \
        # -param step=0,type=label,dof=Tx_Ty_Tz_Sz:step=1,type=im,metric=CC' % sps

        # VERSION 1 : 2 VERTEBRAL LABELS AND FULL SEGMENTATION 
        # JC version - working!
        # run_string = 'cd %s; sct_label_utils -i labels.nii.gz -keep 18,21 -o labels_18_21.nii.gz' % sps
        run_string1 = f'cd {sps}; sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -ldisc labels.nii.gz -c t2 -ofolder vertebral_labels -qc qc\
                       -param step=1,type=seg,algo=centermassrot:step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,iter=3,slicewise=0:step=3,type=im,algo=syn,metric=CC,iter=3,slicewise=0'

        # VERSION2: 1 SPINAL LABEL (CAUDA EQUINA) AND FULL SEGMENTATION
        # REMEBER TO Add the same label on the participant data (terminal)
        # sct_label_utils -i t2.nii.gz -create 33,239,314,99 -o label_caudaequinea.nii.gz'

        run_string2 = f'cd {sps}; sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -ldisc label_caudaequinea.nii.gz -c t2 -ofolder cauda_equina_label -qc qc\
                       -param step=1,type=seg,algo=centermassrot:step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,iter=3,slicewise=0:step=3,type=im,algo=syn,metric=CC,iter=3,slicewise=0'

        ## control for version
        if self.version == 0:
            run_string = run_string0
        elif self.version == 1:
            print("### Info: running version 1 (2 vertebral labels)")
            run_string = run_string1
        elif self.version == 2:
            print("### Info: running version 2 (cauda equina spinal label)")
            run_string = run_string2

        print(run_string)
        os.system(run_string)

    def slice_timing_corr(self):
        # subj_paths = [os.path.join(s, self.func) for s in self.list_subjects]
        subj_paths = self.__check_multisessions('func')
        
        start = time.time()
        print(" ### Info: slice timing correction ...") 

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="multiprocessing")(delayed(self._correct_slice_time)(sps)\
                 for sps in subj_paths)
        
        print("### Info: Slice time correction done in %.3f s" %(time.time() - start ))

    def _correct_slice_time(self, sps):
        
        pathfile = self.func
        if sps.split('/')[-1] == 'func':
            subname = sps.split('/')[-2]
            subpath = "/".join(sps.split("/")[:-1])
            sessions = [""]
            
        else:
            # if multiple sessions
            subname = sps.split('/')[-3]
            subpath = "/".join(sps.split("/")[:-2])

            sessions = ["1","2"]  # this could and should be generalized (now working only for 2 sessions 1 and 2)

        #subname = sps.split('/')[-1]
        for ses in sessions:

            timing_path = os.path.join(subpath,  'dicom', f'{subname}_slice_timings{ses}.txt')

            file_to_realign = os.path.join(subpath, pathfile+ses, 'fmri')
            output_target = os.path.join(subpath, pathfile+ses, 'fmri_st_corr')
            
            cmd = 'slicetimer -i ' + file_to_realign + ' -o ' + output_target + ' -r ' + str(self.TR) + ' -d 3 --ocustom=' + timing_path
            print(cmd)
            os.system(cmd)

        # compute tsnr
        # run_tsnr_fmri = f'sct_fmri_compute_tsnr -i {file_to_realign}.nii.gz -v 0'
        # run_tsnr_fmri_st = f'sct_fmri_compute_tsnr -i {output_target}.nii.gz -v 0' 

        # os.system(f'cd func; {run_tsnr_fmri}; {run_tsnr_fmri_st}')

    def __check_multisessions(self,folder):

        if folder=='func':
            if len(self.func.split('/')) != 1:
                # means that there are multiple sessions within the same func folder
                subj_paths = []
                for s in self.list_subjects:
                    sessions_paths = glob(os.path.join(s, self.func+'*'))
                    subj_paths.extend(sessions_paths)
            else:
                subj_paths = [os.path.join(s, self.func) for s in self.list_subjects]
        elif folder=='physio':
            if len(self.physio.split('/')) != 1:
                # means that there are multiple sessions within the same func folder
                subj_paths = []
                for s in self.list_subjects:
                    sessions_paths = glob(os.path.join(s, self.physio+'*'))
                    subj_paths.extend(sessions_paths)
            else:
                subj_paths = [os.path.join(s, self.physio) for s in self.list_subjects]

        return subj_paths


    def motor_correction(self):

        # subj_paths = [os.path.join(s, self.func) for s in self.list_subjects]
        subj_paths = self.__check_multisessions('func')

        ## (un)comment
        # s = self.list_subjects[0]
        # print(s)
        # subj_paths = [os.path.join(s, self.func)]
        ##
        

        start = time.time()
        print(" ### Info: checking for Mask ...")        
        
        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="multiprocessing")(delayed(self._create_mask)(sub, self.fmriname)\
                 for sub in subj_paths)


        print("### Info: Mask created in %.3f s" %(time.time() - start ))

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="multiprocessing")(delayed(self._moco)(sub, self.fmriname)\
                 for sub in subj_paths)

        print("### Info: Motor correction done in %.3f s" %(time.time() - start ))
        
        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="multiprocessing")(delayed(self._move_processing)(sub, self.fmriname)\
                 for sub in subj_paths)

        ### Generate new file for moco

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="multiprocessing")(delayed(self._generate_new_mocofile)(sub)\
                 for sub in subj_paths)

        return 

    def _create_mask(self, sps, fmriname="fmri"):

        if not os.path.exists(os.path.join(sps, 'Mask', f'mask_{fmriname}.nii.gz')):
            os.makedirs(os.path.join(sps, 'Mask'), exist_ok=True)

            run_string = f'cd {sps}; fslmaths {fmriname}.nii.gz -Tmean {fmriname}_mean.nii.gz; mv {fmriname}_mean.nii.gz Mask; cd Mask; sct_get_centerline -i {fmriname}_mean.nii.gz -c t2;\
            sct_create_mask -i {fmriname}_mean.nii.gz -p centerline,{fmriname}_mean_centerline.nii.gz -size 30mm -o mask_{fmriname}.nii.gz;' 
            print(run_string)
            os.system(run_string)
        else:
            print("### Info: the Mask is already existing!")

    def _moco(self, sps, fmriname="fmri"):
        
        # Kaptan paper they used poly=2
        run_string = f'cd {sps}; sct_fmri_moco -i {fmriname}.nii.gz -m Mask/mask_{fmriname}.nii.gz -x spline -param poly={self.mocopoly} -ofolder Moco; cp Moco/{fmriname}_moco.nii.gz m{fmriname}.nii.gz; cp Moco/{fmriname}_moco_mean.nii.gz m{fmriname}_mean.nii.gz'
        print(run_string)
        os.system(run_string)

    def _move_processing(self, sps, fmriname="fmri"):
        os.makedirs(os.path.join(sps, 'Processing'), exist_ok=True)

        if os.path.exists(os.path.join(sps, 'Mask', f'mask_{fmriname}.nii.gz')):
            run_string = f'cd {sps}; mv fmri* Processing/' # assuming the naming starts with fmri
            os.system(run_string)
            print("Moved fmri file in Processing.")

            # run_string2 = f'cd {sps}; mv Moco/moco_params.tsv Moco/n_moco_params.tsv;\
            #                               mv Moco/moco_params_x.nii.gz Moco/n_moco_params_x.nii.gz \
            #                               mv Moco/moco_params_y.nii.gz Moco/n_moco_params_y.nii.gz'

            # os.system(run_string2)
            # print("Renamed files to not overwrite other runs.")
    
    
    def __load_mocoparams(self, sps, direction):

        vals = nib.load(os.path.join(sps,f'Moco/moco_params_{direction}.nii.gz')).get_fdata()
        newshape = vals.shape[-2:]
        data = vals.reshape(newshape)
        col = np.mean(np.abs(data),0)

        return col

    def _generate_new_mocofile(self, sps):

        ## generate x and y columns
        
        # load moco x/y
        xcol = self.__load_mocoparams(sps,'x')
        ycol = self.__load_mocoparams(sps,'y')        

        mocoxy = zip(xcol,ycol)
        outfilename = sps+'/Moco/new_moco_params.tsv'

        # Write to TSV file
        with open(outfilename, 'w', newline='') as tsvfile:
            # Create a TSV writer
            writer = csv.writer(tsvfile, delimiter='\t')
            # Write header
            writer.writerow(['X', 'Y'])
            # Write data
            writer.writerows(mocoxy)
                

    def func_normalize(self):
        
        # subj_paths = [os.path.join(s, self.func) for s in self.list_subjects]
        subj_paths = self.__check_multisessions('func')
        # (un)comment
        # s = self.list_subjects[12]
        # print(s)
        # subj_paths = [os.path.join(s, self.func)]
        # #
        

        print(" ### Info: Functional Normalization ...") 

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="multiprocessing")(delayed(self._register_multimodal)(sub, self.fmriname)\
                 for sub in subj_paths)

        print("### Info: Functional Normalization done in %.3f s" %(time.time() - start ))

        ### COMMENTING CONCAT BECASUE OF SCT_CONCAT_TRANSFO NOT SUPPORTED ANYMORE
        # print(" ### Info: Concatenate tranformes fmri -> anat & anat->template ...") 
        # # start = time.time()

        # Parallel(n_jobs=self.n_jobs,
        #          verbose=100, 
        #          backend="multiprocessing")(delayed(self._concat_transfo)(sub, self.fmriname)\
        #          for sub in subj_paths)

        # print("### Info: Concatenation tranformation done in %.3f s" %(time.time() - start ))
    
        return

    def _register_multimodal(self, sps,fmriname="fmri"):

        if sps.split('/')[-1] == 'func':
            goback = ''
        else:
            # if multiple sessions
            goback = '../'

        if self.version == 0:
            add = ''
        elif self.version == 1:
            add = '_1'
        elif self.version == 2:
            add = '_2'

        run_string = f'cd {sps}; mkdir -p Normalization; cd Normalization; sct_register_multimodal -i ../m{fmriname}_mean.nii.gz -d {goback}../../{self.anat}/t2.nii.gz \
               -iseg ../Segmentation/{self.mask_fname}.nii.gz -dseg {goback}../../{self.anat}/t2_seg.nii.gz -param step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=seg,algo=affine,metric=MeanSquares,gradStep=0.2:step=3,type=im,algo=syn,metric=CC,iter=5,shrink=2; \
               mv warp_m{fmriname}_mean2t2.nii.gz warp_{fmriname}2anat{add}.nii.gz; mv warp_t22m{fmriname}_mean.nii.gz warp_anat2{fmriname}{add}.nii.gz' 
               
        print(run_string)
        os.system(run_string)

    def _concat_transfo(self, sps, fmriname="fmri"):

        # run_string = 'cd %s; cd Normalization; sct_concat_transfo -w warp_fmri2anat.nii.gz,../../anat/warp_anat2template.nii.gz -o warp_fmri2template.nii.gz -d %s/data/PAM50/template/PAM50_t2.nii.gz;\
        #               sct_concat_transfo -w ../../%s/warp_template2anat.nii.gz,warp_anat2fmri.nii.gz -o warp_template2fmri.nii.gz -d ../mfmri_mean.nii.gz' % (sps, self.SCT_PATH, self.anat)

        # or apply transfo

        if sps.split('/')[-1] == 'func':
            goback = ''
        else:
            # if multiple sessions
            goback = '../'

        if self.version == 0:
            add = ''
            anat_warp = ''
        elif self.version == 1:
            add = '_1'
            anat_warp = 'vertebral_labels/'
        elif self.version == 2:
            add = '_2'
            anat_warp = 'cauda_equina_label/'

        pam50_template_full = os.path.join(self.pam50_path, self.pam50_template+".nii.gz")
        pam50_template_cord = os.path.join(self.pam50_path, self.pam50_maskname+".nii.gz")
        run_string = f'cd {sps}; cd Normalization; \
                      sct_concat_transfo -w warp_{fmriname}2anat{add}.nii.gz {goback}../../{self.anat}/{anat_warp}warp_anat2template.nii.gz -o warp_{fmriname}2template{add}.nii.gz -d {pam50_template_full};\
                      sct_concat_transfo -w {goback}../../{self.anat}/{anat_warp}warp_template2anat.nii.gz warp_anat2{fmriname}{add}.nii.gz -o warp_template2{fmriname}{add}.nii.gz -d ../m{fmriname}_mean.nii.gz'

        # run_string = f'cd {sps}; cd Normalization;\
        #             sct_register_multimodal -i ../m{fmriname}_mean.nii.gz -d {pam50_template_full} -initwarp {goback}../../{self.anat}/{anat_warp}warp_anat2template.nii.gz \
        #             -initwarpinv {goback}../../{self.anat}/{anat_warp}warp_template2anat.nii.gz -iseg ../Segmentation/{self.mask_fname}.nii.gz\
        #             -dseg {pam50_template_cord}../../{self.anat}/t2_seg.nii.gz -param step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=seg,algo=affine,metric=MeanSquares,gradStep=0.2:step=3,type=im,algo=syn,metric=CC,iter=5,shrink=2;'# \
        #             #mv warp_m{fmriname}_mean2template.nii.gz warp_{fmriname}2anat{add}.nii.gz; mv warp_t22m{fmriname}_mean.nii.gz warp_anat2{fmriname}{add}.nii.gz' 
               

        print(run_string)
        os.system(run_string)
    

    def prepare_physio(self):
        subj_paths = self.__check_multisessions('physio')

        # subj_paths = [os.path.join(s, self.physio) for s in self.list_subjects]
        # ######
        # subj_paths = [os.path.join(s, self.physio) for s in self.list_subjects if s.split('/')[-1] in  \
        #               ['LU_GL']]
                      
        # print(subj_paths)
        # ######

        # (un)comment
        # s = self.list_subjects[2]
        # print(s)
        # subj_paths = [os.path.join(s, self.physio)]
        #

        print(" ### Info: Converting text file ...") 

        start = time.time()
        
        ### NEW : not parallel anymore:
        ### This parallel not always works becasue the processes are left there and
        ## the txt file is not generated eventually

        # Parallel(n_jobs=self.n_jobs,
        #          verbose=100,
        #          backend="multiprocessing")(delayed(self._convert_txt_filt)(sub)\
        #          for sub in subj_paths)

        for sub in subj_paths:
            self._convert_txt_filt(sub)

        print("### Info: File conversion done in %.3f s" %(time.time() - start ))
        return

    def __get_mat_info(self,sub):
        matstructfile = glob(os.path.join(sub, '*.mat'))[0]
        matstruct = loadmat(matstructfile)
        data = matstruct['data']
        FS = 1/matstruct['isi']*1000
        isi = matstruct['isi']

        return matstructfile, data, FS[0][0]

    def _convert_txt_filt(self, sub):

        os.chdir(sub)   
        # matstructfile = glob(os.path.join(os.getcwd(), '*.mat'))[0]
        # matstruct = loadmat(matstructfile)
        # data = matstruct['data']
        # FS = 1/matstruct['isi']*1000
        
        matstructfile, data, FS = self.__get_mat_info(sub)

        if self.filter_cardiac:
            cof = self.cof
            if len(self.physio.split('/')) != 1:
                sub_oi = sub.split('/')[-3].split('_')[0]
            else:
                sub_oi = sub.split('/')[-2]
            # exception subjects for filter for Lumbar and Rob's data
            if sub_oi in ['LU_NK','LU_MD', 'LU_MP','LU_NG','LU_SA','LU_SM','LU_VS']:
                cof = 1
            elif sub_oi == 'LU_GL':
                cof = 3
            elif sub_oi == 'LU_SL':
                cof = 1.5
            elif sub_oi in ['sID-38','sID-42','sID-44','sID-50']:
                cof = 2 
            elif sub_oi == 'sID-51':
                cof = 3
            elif sub_oi == 'sID-52':
                cof = 1.5
            
            Wn = (cof*2)/FS
            if Wn > 1.0:
                Wn = 0.99
            B,A = butter(3,Wn,'low')
            data[:,int(self.pnm_columns['cardiac'])-1] = filtfilt(B,A,data[:,int(self.pnm_columns['cardiac'])-1])

            data = np.array(data)

        np.savetxt(matstructfile[:-3]+'txt', data, fmt='%.4f', delimiter='\t')

    def pnm_stage1(self):
        
        subj_paths = self.__check_multisessions('physio')
        # subj_paths = [os.path.join(s, self.physio) for s in self.list_subjects]
        # ######
        # subj_paths = [os.path.join(s, self.physio) for s in self.list_subjects if s.split('/')[-1] in  \
        #               ['LU_GL']]
                      
        # print(subj_paths)
        ######
        

        # # (un)comment
        # s = self.list_subjects[4]
        # print(s)
        # subj_paths = [os.path.join(s, self.physio)]
        # #

        print("### Info: Physiological preparetion ...") 

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="loky")(delayed(self._fsl_pnm_stage1)(sub)\
                 for sub in subj_paths)

        print("### Info: Physiological preparetion done in %.3f s" %(time.time() - start ))
        return

    def _fsl_pnm_stage1(self, sps):

        # cardiac_string = ""
        # resp_string = ""

        # if self.pnm1_card:
        #     cardiac_string = f"--smoothcard=0.3 --cardiac={self.pnm_columns['cardiac']}"
        #     print("### Info: running cardiac PNM ... ")
        # if self.pnm1_resp:
        #     resp_string = f"--smoothresp=0.1 --resp={self.pnm_columns['resp']}"
        #     print("### Info: running respiratory PNM ... ")

        if 'trigger' in self.pnm_columns:
            trigger_opt = f"--trigger={self.pnm_columns['trigger']}"
        else:
            trigger_opt = ''

        if len(self.physio.split('/')) != 1:
            subjname = sps.split('/')[-3].split('_')[0]
        else:
            subjname = sps.split('/')[-2]
        os.chdir(sps) 
        _, _,  FS = self.__get_mat_info(sps)

        # fslFixText / popp
        # you can play with smoothcard param choosing 0.2-0.5
        run_string = f"cd {sps};  {self.FSL_PATH}bin/fslFixText ./{subjname}.txt ./{subjname}_input.txt; {self.FSL_PATH}bin/pnm_stage1 -i ./{subjname}_input.txt\
                       -o {subjname} -s {str(int(FS))} --tr={str(self.TR)} --resp={self.pnm_columns['resp']}  --cardiac={self.pnm_columns['cardiac']}\
                       --smoothresp=0.1 --smoothcard=0.3 {trigger_opt}"


        print(run_string)
        os.system(run_string)

    def pnm_stage2(self):

        subj_paths = self.__check_multisessions('physio')
        # subj_paths = [os.path.join(s, self.physio) for s in self.list_subjects]
        
        # ######
        # subj_paths = [os.path.join(s, self.physio) for s in self.list_subjects if s.split('/')[-1] in  \
        #               ['LU_GL']]
                      
        # print(subj_paths)
        ######
        
        ## (un)comment
        # s = self.list_subjects[0]
        # print(s)
        # subj_paths = [os.path.join(s, self.physio)]
        ##

        print("### Info: Physiological preparetion ...") 

        start = time.time()

        if self.mode == 'auto':

            Parallel(n_jobs=self.n_jobs,
                     verbose=100,
                     backend="multiprocessing")(delayed(self._check_peaks_car_persub)(sub)\
                     for sub in subj_paths)
        else:
            # manual check of the peaks
            print(" ### SUBJECT %s chosen for single caridac peaks identification! " % self.mode)
            self._check_peaks_car_persub(self.mode, auto=False)

        _send_message_telegram("PNM!")
        print("### Info: Physiological preparetion done in %.3f s" %(time.time() - start ))
        return

    def _check_peaks_car_persub(self, sps, auto=True):

        if auto:
            sub_path = sps
            if len(self.physio.split('/')) != 1:
                subname = sps.split('/')[-3].split('_')[0]
            else:
                subname = sps.split('/')[-2]

        else:
            sub_path = os.path.join(self.parent_path, sps, self.physio)
            subname = sps
            # not used because automatically used
            # seq_card = np.loadtxt(os.path.join(sub_path, subname+'_card.txt'))
            # seq_card = (np.round(seq_card*FS)).astype(int)

        os.chdir(sps)
        
        _, _, FS = self.__get_mat_info(sps)
        print(f"### Info: FS = {FS}")

        seq_input = np.loadtxt(os.path.join(sub_path, subname+'_input.txt'))
        col_car = int(self.pnm_columns["cardiac"])-1  # -1 for python indexing


        ### In Rob's dataset triggers are already found and signals have been adjusted
        ### so if no trigger is specified in the pnm_columns structure in config file,
        ### then this doesn't need to be run and only the cardiac signal can be found

        if 'trigger' in self.pnm_columns:

            # Find triggers 
            col_tr = int(self.pnm_columns["trigger"])-1   # in python -1 indexing
            triggers = np.where(seq_input[:,col_tr]==TRG)[0]
            card_signal = seq_input[triggers[0]:,col_car]/YFACTOR  # /10 is more for visualization, factor doesn't change or 100
                                                                   # it depends on data so check 
            
        else:
            valmax = str(np.max(seq_input[:,col_car]))
            factor = len(valmax)-1
            card_signal = seq_input[:,col_car]/(10**factor)

        indices = find_peaks(card_signal)[0]

        fig = self.__interactive_plot(card_signal, indices)
        fig.write_html(os.path.join(sub_path, '%s_hr_peaks.html') % subname)
        
        auto_detect = indices/FS
        np.savetxt(os.path.join(sub_path, subname+'_card_auto.txt'), auto_detect, delimiter='\n', fmt='%.3f') 
        
    def __interactive_plot(self, card_signal, auto_idx):

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=card_signal,
            mode='lines',
            name='Original Plot'
        ))

        if len(auto_idx) <= 2000:
            # plot lines per peak
            lines = dict(zip([str(i) for i in range(len(auto_idx))], auto_idx))

            # add lines using absolute references
            for k in lines.keys():

                fig.add_shape(type='line',
                            # yref="y",
                            # xref="x",
                            x0=lines[k],
                            y0=-0.5,  # 0.1
                            x1=lines[k],
                            y1=0.5,   # 0.1 
                            line=dict(color='red', width=1))
        else:
            # otherwise ad a marker with a red cross
            fig.add_trace(go.Scatter( 
                x=auto_idx,
                y=[card_signal[j] for j in auto_idx],
                opacity=0.8,
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='cross'
                ),
                name='Automatic Peaks'
            ))

        return fig

    def generate_evs(self):
        subj_paths = self.__check_multisessions('physio')
        # subj_paths = [os.path.join(s, self.physio) for s in self.list_subjects]
        # subj_paths = [os.path.join(s, self.physio) for s in self.list_subjects if s.split('/')[-1] != 'LU_AT']
        # # (un)comment
        # s = self.list_subjects[0]
        # print(s)
        # subj_paths = [os.path.join(s, self.physio)]
        # #

        print("### Info: Generate EVS ...") 

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="multiprocessing")(delayed(self._fsl_pnm_evs)(sub, self.fmriname)\
                 for sub in subj_paths)

        print("### Info: EVS generation done in %.3f s" %(time.time() - start ))
        return 

    def _fsl_pnm_evs(self, sub, fmriname="fmri"):

        if len(self.physio.split('/')) != 1:
            subname = sub.split('/')[-3].split('_')[0]
            
        else:
            subname = sub.split('/')[-2]

        func_path = sub.replace(self.physio,self.func)        
        #subname = sub.split('/')[-2]
        out_reg = subname
        if 'st' in fmriname.split('_'): 
            # in the case fmri name is fmri_st or some other condition, the condition will be written 
            out_evs = subname + '_st'
            out_reg += "_st"
        else:
            out_evs = subname
            

        if self.mode == 'auto':
            auto_mode = '_auto'
        else:
            auto_mode = ''

        if 'st' in fmriname.split('_'): 
            sliceorder = ""
        else:
            sliceorder = "--sliceorder=up"

        if self.pnm1_card == True and self.pnm1_resp == False:
            out_evs += "_cardonly"
            out_reg += "_cardonly"
            
        elif self.pnm1_card == False and self.pnm1_resp == True:
            out_evs += "_responly"
            out_reg += "_responly"

        if self.csf_mask == True:
            # if there is a mask of the CSF only (mask_csf.nii) : it can be manually created or using 
            # fslmaths subtracting the mask of only the spinal cord from the mask of CSF+cord 
            csfmask = f'--csfmask="{func_path}/Segmentation/{self.mask_csf_name}.nii.gz"'
            evs_names = f"{out_evs}_csf_ev0"
            out_evs += "_csf"
        else :
            # if no csf mask
            csfmask = ''
            evs_names = f"{out_evs}_ev0" ### for Kinany2020: f"{out_evs}ev0" 

        # init cardiac/respirratory
        ocv = 0
        multcv = 0 
        orv = 0
        multrv = 0
        if self.pnm1_card:
            ocv = 4
            multcv = 2
        if self.pnm1_resp:
            orv = 4
            multrv = 2

        run_string = f'cd {sub}; {self.FSL_PATH}bin/pnm_evs -i {func_path}/m{fmriname}.nii.gz -c {subname}_card{auto_mode}.txt \
                    --oc={ocv} -r {subname}_resp.txt --or={orv} --multc={multcv} --multr={multrv} -o {sub}/{out_evs}_ --tr={self.TR} \
                    {csfmask} {sliceorder} --slicedir=z' 

        print(run_string)
        os.system(run_string)

        run_string2 = f'cd {sub}; ls -1 `{self.FSL_PATH}bin/imglob -extensions {sub}/{evs_names}*` > {sub}/{out_evs}_evlist.txt'

        print(run_string2)
        os.system(run_string2)


    def apply_denoising(self):
        
        # subj_paths = [os.path.join(s, self.func) for s in self.list_subjects]
        subj_paths = self.__check_multisessions('func')

        # # (un)comment
        # s = self.list_subjects[0]
        # print(s)
        # subj_paths = [os.path.join(s, self.func)]
        # #

        print("### Info: Generate denoising regressors ...") 

        print("SPlitting the images...")
        # add = ''
        # if 'st' in self.fmriname:
        #     add = '_st'
        
        # for sps in subj_paths:
        #     #if len(os.listdir(os.path.join(sps,f'splitted_mfmri{add}'))) != self.nslices+1 : 
        #     ## +1 for the full volume reconstruction

        #     os.makedirs(os.path.join(sps,f'splitted_mfmri{add}'),exist_ok=True)
        #     self.split_img_slices(sps)

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                verbose=100,
                backend="multiprocessing")(delayed(self._clean_signals)(sub, self.fmriname)\
                for sub in subj_paths)

        print("### Info: Denoising done in %.3f s" %(time.time() - start ))
        
        _send_message_telegram(self.denoising_regs)

        return 

    def _compcor_persub(self, sub, fmriname="fmri"):

        outname = "regressors"
        print("### Info: running CompCor... ")
        os.chdir(sub)

        ccinterface = CompCor()
        ccinterface.inputs.realigned_file = f"m{fmriname}.nii.gz"
        ccinterface.inputs.mask_files = os.path.join("Segmentation", self.mask_csf_name+".nii.gz")
        ccinterface.inputs.num_components = 5
        ccinterface.inputs.pre_filter = False
        ccinterface.inputs.repetition_time = self.TR
        if 'st' in fmriname.split('_'): 
            ccinterface.inputs.components_file = f"{outname}_compcor_st.txt"
        else:
            ccinterface.inputs.components_file = f"{outname}_compcor.txt"
        ccinterface.run()

    def __count_lines_txt(self, filepath):
        return sum(1 for _ in open(filepath, 'r'))

    def __create_confounds_pnm(self,sps,filename):
    
        filepath = os.path.join(sps,filename)
        nc = self.__count_lines_txt(filepath)
        # print("HEEREEEE subject:", sps)
        # populate the list of Nc regressors per slice
        confounds = np.zeros((self.nslices, self.tpoints, nc))
        with open(os.path.join(sps,filename)) as fp:
            # go back to first line
            fp.seek(0)
            for c, line in enumerate(fp):
                line = line.strip()
                img = nib.load(line)
                f_img_data = img.get_fdata()
                
                tsignalslices = f_img_data.reshape([f_img_data.shape[-2], f_img_data.shape[-1]])
                confounds[:,:,c] = tsignalslices

        # # save txt file
        # np.savetxt(os.path.join(sps,outname),confounds, delimiter='\t')
        # fp.close()
        return confounds 

    def split_img_slices(self,sps):

        add = ''
        if 'st' in self.fmriname:
            add = '_st'
            
        spfolder = os.path.join(sps,f'splitted_mfmri{add}')
        os.system(f"rm -rf {spfolder}")
        os.makedirs(os.path.join(sps,f'splitted_mfmri{add}'),exist_ok=True)
        
        if len(os.listdir(os.path.join(sps,f'splitted_mfmri{add}'))) != self.nslices:
            # if there are no files, split the fmri per slice
            for i in range(self.nslices):
                fimg_slice = self.fmri_data[:,:,i,:]
                fimg_slice = fimg_slice[:,:,np.newaxis,:]
                
                img_nii = nib.Nifti1Image(fimg_slice,self.fmri_affine, self.fmri_header)
                outpath = os.path.join(sps, f'splitted_mfmri{add}',f'm{self.fmriname}_{i}.nii.gz')
                nib.save(img_nii, outpath)

        return


    def _clean_signals(self, sps, fmriname="fmri"):

        # save needed variable for timepoints
        fmri = nib.load(os.path.join(sps,f"m{fmriname}.nii.gz"))
        self.fmri_data = fmri.get_fdata()
        self.fmri_affine = fmri.affine
        self.fmri_header = fmri.header
        self.tpoints = fmri.shape[-1]
        self.nslices = fmri.shape[-2]

        print("self.tpoints: ", self.tpoints)
        print("self.nslices: ", self.nslices) 

        os.makedirs(os.path.join(sps, 'Nuisance'), exist_ok=True)
        ## Initialize denoised output name 
        denoised_name = f"m{fmriname}_denoised"
        add_outname = ""
         # empty dict that will have the confounds
        nuisance_mat_dict = {'pnm': None,
                             'csfonly': None,
                             'moco': None,
                             'compcor': None
                             }

        print("### Info: generating nuisance regressors ...")

        ## Prepare filenames according to variant especially for slice timing (st)
        if len(self.func.split('/')) != 1:
            subname = sps.split('/')[-3].split('_')[0]
        else:
            subname = sps.split('/')[-2]

        out_reg = "regressors_evlist"
        add = ""
        ## Start st variat