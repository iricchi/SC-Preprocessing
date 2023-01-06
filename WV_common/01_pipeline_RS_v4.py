# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
IMPORTANT NOTE: The denoising with feat works only on miplabsrv4 because fsl5 is there only.

This pipeline is used mainly for RESTING STATE. 

Order of the options (recommended to follow):

STEPS :

    // = could be done in parallel since one is one anat and the other on func
 
    0) Segmentation with SCT and labels generation (outside the pipeline)
    
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
import requests

def _send_message_telegram(whatever):

    TOKEN = "5979998311:AAGl9Did2fwB_1rEd0RHA496Qox6xLghrOM"
    chat_id = "626146439"
    message = f"Pipeline: I'm done with {whatever}!"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    print(requests.get(url).json()) # this sends the message


class PreprocessingRS(object):
    """PreprocessingRS performs the STEP1 for the whole Pipeline in RS.

    NOTE: segmentation and labelling are done separately and should be visually evaluated!
    
    Commands used:
    sct_deepseg_sc with viewer initialization
    sct_deepseg_sc -i t2.nii.gz -c t2 -centerline viewer

    ### Lumbar
    sct_deepseg -task seg_lumbar_sc_t2w -i t2.nii.gz 
    sct_label_utils -i t2.nii.gz -create 32,257,374,99 -o label_caudaequinea.nii.gz
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

    NOTES: 
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

        self.anat_norm = False
        self.slice_timing = False
        self.moco = False
        self.func_norm = False
        self.mask_fname = "mask_sc"
        self.mask_csf_name = "mask_csf"
        self.pnm0_preptxt = False   # step 0: prepare physio rec (txt file and filter)
        self.pnm1_stage1 = False   # first step: pnm_stage 1
        self.pnm2_peaks = False   # second step: check peaks of cardiac sig
        self.pnm3_gen_evs = False   # third step: generate evs 
        self.cof = 2  # default filter parameter for pnm0
        self.filter = False   # by default don't filter data
        self.mode = ''
        self.fsessions_name = ''
        self.csf_mask = False  # default / in cervical should be true
        self.denoising = False
        self.denoising_regs = [] # can contain compcor (PCA-like additional physio noise regs)
        self.normalization = False
        self.smoothing = False
        self.prep_for_ta = False

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
                # initialize or overwrite value
                setattr(self, key, params[key])

        # assign the list of subjects variable
        self.list_subjects = glob(os.path.join(self.parent_path, self.data_root+'*'))

        subj_found = [sub.split('/')[-1] for sub in self.list_subjects]
        print("Subjects found:", subj_found)
        
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

        start = time.time()
        print(" ### Info: slice timing correction ...") 

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="multiprocessing")(delayed(self._correct_slice_time)(subpath)\
                 for subpath in self.list_subjects)
        
        print("### Info: Slice time correction done in %.3f s" %(time.time() - start ))

    def _correct_slice_time(self, subpath):
        
        subname = subpath.split('/')[-1]
        timing_path = os.path.join(subpath,  'dicom', f'{subname}_slice-timings.txt')

        file_to_realign = os.path.join(subpath, 'func', f'{self.fmriname}')
        output_target = os.path.join(subpath, 'func', 'fmri_st_corr')

        cmd = 'slicetimer -i ' + file_to_realign + ' -o ' + output_target + ' -r ' + str(self.TR) + ' -d 3 --ocustom=' + timing_path
        os.system(cmd)

        # compute tsnr
        run_tsnr_fmri = f'sct_fmri_compute_tsnr -i {self.fmriname}.nii.gz -v 0'
        run_tsnr_fmri_st = 'sct_fmri_compute_tsnr -i fmri_st_corr.nii.gz -v 0' 

        os.system(f'cd func; {run_tsnr_fmri}; {run_tsnr_fmri_st}')

    def motor_correction(self):

        # # if there are multiple sessions (e.g. with meditation data)
        # if len(self.fsessions_name) > 0:
        #     subj_paths = [ses for s in self.list_subjects for ses in glob(os.path.join(s, self.func, self.fsessions_name+'*'))]

        # else:
        subj_paths = [os.path.join(s, self.func) for s in self.list_subjects]

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

        return 

    def _create_mask(self, sps, fmriname="fmri"):

        if not os.path.exists(os.path.join(sps, 'Mask', f'mask_{fmriname}.nii.gz')):
            os.makedirs(os.path.join(sps, 'Mask'), exist_ok=True)

            run_string = f'cd %s; fslmaths {fmriname}.nii.gz -Tmean {fmriname}_mean.nii.gz; mv {fmriname}_mean.nii.gz Mask; cd Mask; sct_get_centerline -i {fmriname}_mean.nii.gz -c t2;\
            sct_create_mask -i {fmriname}_mean.nii.gz -p centerline,{fmriname}_mean_centerline.nii.gz -size 30mm -o mask_{fmriname}.nii.gz;' % sps
            print(run_string)
            os.system(run_string)
        else:
            print("### Info: the Mask is already existing!")

    def _moco(self, sps, fmriname="fmri"):

        run_string = f'cd {sps}; sct_fmri_moco -i {fmriname}.nii.gz -m Mask/mask_{fmriname}.nii.gz -x spline -param poly=0 -ofolder Moco; cp Moco/{fmriname}_moco.nii.gz m{fmriname}.nii.gz; cp Moco/{fmriname}_moco_mean.nii.gz m{fmriname}_mean.nii.gz'
        print(run_string)
        os.system(run_string)

    def _move_processing(self, sps, fmriname="fmri"):
        os.makedirs(os.path.join(sps, 'Processing'), exist_ok=True)

        if os.path.exists(os.path.join(sps, 'Mask', f'mask_{fmriname}.nii.gz')):
            run_string = f'cd {sps}; mv {fmriname}.nii.gz Processing/{fmriname}.nii.gz'
            os.system(run_string)
            print("Moved fmri file in Processing.")
            if not self.slice_timing:
                run_string2 = f'cd {sps}; mv Moco/moco_params.tsv Moco/n_moco_params.tsv;\
                                          mv Moco/moco_params_x.nii.gz Moco/n_moco_params_x.nii.gz \
                                          mv Moco/moco_params_y.nii.gz Moco/n_moco_params_y.nii.gz'

                os.system(run_string2)
                print("Renamed files to not overwrite other runs.")


    def func_normalize(self):
        
        subj_paths = [os.path.join(s, self.func) for s in self.list_subjects]
        
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
        print(" ### Info: Concatenate tranformes fmri -> anat & anat->template ...") 
        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100, 
                 backend="multiprocessing")(delayed(self._concat_transfo)(sub, self.fmriname)\
                 for sub in subj_paths)

        print("### Info: Concatenation tranformation done in %.3f s" %(time.time() - start ))
    
        return

    def _register_multimodal(self, sps,fmriname="fmri"):

        if self.version == 0:
            add = ''
        elif self.version == 1:
            add = '_1'
        elif self.version == 2:
            add = '_2'

        run_string = f'cd {sps}; mkdir -p Normalization; cd Normalization; sct_register_multimodal -i ../m{fmriname}_mean.nii.gz -d ../../{self.anat}/t2.nii.gz \
               -iseg ../Segmentation/{self.mask_fname}.nii.gz -dseg ../../{self.anat}/t2_seg.nii.gz -param step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=seg,algo=affine,metric=MeanSquares,gradStep=0.2:step=3,type=im,algo=syn,metric=CC,iter=5,shrink=2; \
               mv warp_m{fmriname}_mean2t2.nii.gz warp_{fmriname}2anat{add}.nii.gz; mv warp_t22m{fmriname}_mean.nii.gz warp_anat2{fmriname}{add}.nii.gz' 
               
        print(run_string)
        os.system(run_string)

    def _concat_transfo(self, sps, fmriname="fmri"):

        # run_string = 'cd %s; cd Normalization; sct_concat_transfo -w warp_fmri2anat.nii.gz,../../anat/warp_anat2template.nii.gz -o warp_fmri2template.nii.gz -d %s/data/PAM50/template/PAM50_t2.nii.gz;\
        #               sct_concat_transfo -w ../../%s/warp_template2anat.nii.gz,warp_anat2fmri.nii.gz -o warp_template2fmri.nii.gz -d ../mfmri_mean.nii.gz' % (sps, self.SCT_PATH, self.anat)

        # or apply transfo
        if self.version == 0:
            add = ''
        elif self.version == 1:
            add = '_1'
            anat_warp = 'vertebral_labels'
        elif self.version == 2:
            add = '_2'
            anat_warp = 'cauda_equina_label'

        run_string = f'cd {sps}; cd Normalization; \
                      sct_concat_transfo -w warp_{fmriname}2anat{add}.nii.gz ../../{self.anat}/{anat_warp}/warp_anat2template.nii.gz -o warp_{fmriname}2template{add}.nii.gz -d {self.SCT_PATH}data/PAM50/template/PAM50_t2.nii.gz;\
                      sct_concat_transfo -w ../../{self.anat}/{anat_warp}/warp_template2anat.nii.gz warp_anat2{fmriname}{add}.nii.gz -o warp_template2{fmriname}{add}.nii.gz -d ../m{fmriname}_mean.nii.gz'

        print(run_string)
        os.system(run_string)

    def prepare_physio(self):

        subj_paths = [os.path.join(s, self.physio) for s in self.list_subjects]
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

        if self.filter:
            cof = self.cof
            sub_oi = sub.split('/')[-2]
            # exception subjects for filter
            if sub_oi in ['LU_NK','LU_MD', 'LU_MP','LU_NG','LU_SA','LU_SM','LU_VS']:
                cof = 1
            elif sub_oi == 'LU_GL':
                cof = 3
            elif sub_oi == 'LU_SL':
                cof = 1.5

            
            Wn = (cof*2)/FS
            if Wn > 1.0:
                Wn = 0.99
            B,A = butter(3,Wn,'low')
            data[:,int(self.pnm_columns['cardiac'])-1] = filtfilt(B,A,data[:,int(self.pnm_columns['cardiac'])-1])

            data = np.array(data)

        np.savetxt(matstructfile[:-3]+'txt', data, fmt='%.4f', delimiter='\t')

    def pnm_stage1(self):

        subj_paths = [os.path.join(s, self.physio) for s in self.list_subjects]
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
                 backend="multiprocessing")(delayed(self._fsl_pnm_stage1)(sub)\
                 for sub in subj_paths)

        print("### Info: Physiological preparetion done in %.3f s" %(time.time() - start ))
        return

    def _fsl_pnm_stage1(self, sps):

        subjname = sps.split('/')[-2]
        os.chdir(sps) 
        _, _,  FS = self.__get_mat_info(sps)
        # fslFixText / popp

        # you can play with smoothcard param choosing 0.2-0.5

        run_string = 'cd %s; %sbin/fslFixText ./%s.txt ./%s_input.txt; %sbin/pnm_stage1 -i ./%s_input.txt -o %s -s %s --tr=%s \
        --smoothcard=0.3 --smoothresp=0.1 --resp=%s --cardiac=%s --trigger=%s' % (sps,
                                                                                  self.FSL_PATH, 
                                                                                  subjname,
                                                                                  subjname,
                                                                                  self.FSL_PATH,
                                                                                  subjname,
                                                                                  subjname,                                                                 
                                                                                  str(FS),
                                                                                  str(self.TR),  
                                                                                  self.pnm_columns['resp'],
                                                                                  self.pnm_columns['cardiac'],  
                                                                                  self.pnm_columns['trigger'])                                                                               

        #/usr/local/fsl/bin/popp -i ./sub-001_input.txt -o ./Stim -s 2000 --tr=2.5 --smoothcard=0.3 --smoothresp=0.1 --resp=1 --cardiac=2 --trigger=3

        print(run_string)
        os.system(run_string)

    def pnm_stage2(self):

        subj_paths = [os.path.join(s, self.physio) for s in self.list_subjects]
        
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

        print("### Info: Physiological preparetion done in %.3f s" %(time.time() - start ))
        return

    def _check_peaks_car_persub(self, sps, auto=True):

        if auto:
            sub_path = sps
            subname = sps.split('/')[-2]
        else:
            sub_path = os.path.join(self.parent_path, sps, self.physio)
            subname = sps
        os.chdir(sps)
        _, _, FS = self.__get_mat_info(sps)
        print(f"### Info: FS = {FS}")

        seq_input = np.loadtxt(os.path.join(sub_path, subname+'_input.txt'))
        seq_card = np.loadtxt(os.path.join(sub_path, subname+'_card.txt'))
        
        seq_card = (np.round(seq_card*FS)).astype(int)

        # Find triggers 
        col_tr = int(self.pnm_columns["trigger"])-1   # in python -1 indexing
        triggers = np.where(seq_input[:,col_tr]==5)[0]
        col_car = int(self.pnm_columns["cardiac"])-1 
        card_signal = seq_input[triggers[0]:,col_car]/10

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
                            y0=-0.1,
                            x1=lines[k],
                            y1=0.1,
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
        subj_paths = [os.path.join(s, self.physio) for s in self.list_subjects]
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

        subname = sub.split('/')[-2]
        if len(fmriname.split('_')) != 1:
            # in the case fmri name is fmri_st or some other condition, the condition will be written 
            out_evs = subname + '_st'
        else:
            out_evs = subname

        if self.mode == 'auto':
            auto_mode = '_auto'
        else:
            auto_mode = ''

        if self.slice_timing :
            sliceorder = ""
        else:
            sliceorder = "--sliceorder=up"


        if self.csf_mask == True:
            # if there is a mask of the CSF only (mask_csf.nii) : it can be manually created or using 
            # fslmaths subtracting the mask of only the spinal cord from the mask of CSF+cord 
            
            run_string = f'cd {sub}; {self.FSL_PATH}bin/pnm_evs -i ../{self.func}/m{fmriname}.nii.gz -c {subname}_card{auto_mode}.txt -r {subname}_resp.txt -o {sub}/{out_evs}_csf_ --tr={self.TR} --oc=4 --or=4 --multc=2 \
            --multr=2 --csfmask="../{self.func}/Segmentation/{self.mask_csf_name}.nii.gz" {sliceorder} --slicedir=z' 
            evs_names = f"{out_evs}_csf_ev0"
            out_evs += "_csf"
        else :
            # if no csf mask
            run_string = f'cd {sub}; {self.FSL_PATH}bin/pnm_evs -i ../{self.func}/m{fmriname}.nii.gz -c {subname}_card{auto_mode}.txt -r {subname}_resp.txt -o {sub}/{out_evs}_ --tr={self.TR} --oc=4 --or=4 --multc=2 \
            --multr=2 {sliceorder} --slicedir=z' 
            evs_names = f"{out_evs}_ev0"

        print(run_string)
        os.system(run_string)

        run_string2 = f'ls -1 `{self.FSL_PATH}bin/imglob -extensions {sub}/{evs_names}*` > {sub}/{out_evs}_evlist.txt'

        print(run_string2)
        os.system(run_string2)
        

    def apply_denoising(self):
        subj_paths = [os.path.join(s, self.func) for s in self.list_subjects]
        # subj_paths = [os.path.join(s, self.func) for s in self.list_subjects if s.split('/')[-1] == 'LU_VG']
        # print(subj_paths)

        # # (un)comment
        # s = self.list_subjects[0]
        # print(s)
        # subj_paths = [os.path.join(s, self.func)]
        # #

        print("### Info: Generate denoising regressors ...") 

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                verbose=100,
                backend="multiprocessing")(delayed(self._fsl_feat_regressors)(sub, self.fmriname)\
                for sub in subj_paths)

        print("### Info: Denoising done in %.3f s" %(time.time() - start ))
        
        _send_message_telegram(self.denoising_regs)

        return 

    def _compcor_persub(self, sub, fmriname="mfmri"):

        outname = "regressors"
        if len(fmriname.split('_')) != 1:
            outname += "_st"

        print("### Info: running CompCor... ")
        os.chdir(sub)

        ccinterface = CompCor()
        ccinterface.inputs.realigned_file = f"{fmriname}.nii.gz"
        ccinterface.inputs.mask_files = os.path.join("Segmentation", self.mask_csf_name+".nii.gz")
        ccinterface.inputs.num_components = 5
        ccinterface.inputs.pre_filter = False
        ccinterface.inputs.repetition_time = self.TR
        ccinterface.inputs.components_file = f"{outname}_compcor.txt"

        ccinterface.run()

    def _fsl_feat_regressors(self, sps, fmriname="fmri"):

        #### Potential cases are 
        # 1) only outliers
        # 2) only moco
        # 3) only PNM (vox)
        # 4) Only CSF
        # 5) Only compcor

        # And all possible combinations
        # in particular older preprocessing version was 1+2+3
        # then we can add also 4 for csf
        # But after confrontation also compcor brings good results. 

        # Make directory for Nuisance (outliers)
        os.makedirs(os.path.join(sps, 'Nuisance'), exist_ok=True)

        print("### Info: generating nuisance regressors ...")

        ## Start st variation
        # prepare filenames according to variant especially for slice timing (st)
        subname = sps.split('/')[-2]
        out_reg = "regressors_evlist"
        noise_reg_out = "noise_regression"
        nuisance_txt = 'nuisance'
        txtyn = "0"
        outliersname = "outliers"
        
        if len(fmriname.split('_')) != 1:
            # in the case fmri name is fmri_st or some other condition, the condition will be written 
            subname += "_st"
            out_reg += "_st"
            noise_reg_out += "_st"
            nuisance_txt += "_st"

            moco_params_name = "moco_params"
            moco = "moco_st"
            outliersname += "_st"
        else:
            # the moco params are named n_moco_params if no slice timing
            # while simply moco_params if with slice timing correctin
            moco_params_name = "n_moco_params"   
            moco = "moco"     

        moco_params_name = "n_moco_params"

        if self.csf_mask:
            subname = subname+"_csf"
            out_reg = out_reg+"_csf"
            noise_reg_out = noise_reg_out+"_csf"
        ## end st

        ## Initialize denoised output name 
        denoised_name = "denoised"
        add_outname = ""

        ## start PNM regressors
        # generate usefull names for PNM
        physiopath = sps.split('/')[:-1]
        # physiopath = os.path.join('/',*physiopath, self.physio)
        phypath = os.path.join(*physiopath)
        physiopath = os.path.join('/',phypath, self.physio)

        if "pnm" in self.denoising_regs:
            print("### Info: adding PNM regressors ...")
            run_string = f'cp {physiopath}/{subname}_evlist.txt {sps}/{out_reg}.txt;'
            print(run_string)
            os.system(run_string)
            add_outname += "_pnm"
        if "csf" in self.denoising_regs and "pnm" not in self.denoising_regs:
            # copy only last line of the regressors (which sould be csf)
            run_string = f'tail -n -1 {physiopath}/{subname}_evlist.txt >> {sps}/{out_reg}.txt;'
            print(run_string)
            print("### Info: taking only the CSF regressor")
            os.system(run_string)
            add_outname += "_csf"
        elif "csf" in self.denoising_regs and "pnm" in self.denoising_regs:
            add_outname += "_csf"

        if "pnm" not in self.denoising_regs and  "csf" not in self.denoising_regs:
            pnmpaths = ''
        else:
            pnmpaths = os.path.join(sps,f"{out_reg}.txt")
       
        ## end PNM regressors

        ## start voxel-wise txt (outliers, moco and compcor)
        # by default outliers, moco and compcor empty
        add_outliers = ''
        add_moco     = ''
        add_compcor  = ''

        if "outliers" in self.denoising_regs:
            print("### Info: adding outliers regressors...")
            # use regressors from the outliers file if existing
            if not os.path.exists(os.path.join(sps,"outliers.png")):
                run_string = f'cd {sps}; fsl_motion_outliers -i m{fmriname}.nii.gz -o Nuisance/{subname}_outliers.txt —m Segmentation/{self.mask_fname}.nii.gz -p Nuisance/{subname}_outliers.png --dvars --nomoco;'
                print(run_string)
                os.system(run_string)
            
            # if outliers txt file exist then 
            if os.path.exists(os.path.join(sps, f"Nuisance/{subname}_outliers.txt")):
                run_string = f"cd {sps}; sed -e 's/   /\t/g' Nuisance/{subname}_outliers.txt > Nuisance/outliers_tmp.txt;\
                               mv Nuisance/outliers_tmp.txt Nuisance/{outliersname}.txt"
                print(run_string)
                os.system(run_string)
                add_outliers = f"Nuisance/{outliersname}.txt"
                txtyn = "1"
            add_outname += "_outl"
        
        if "moco" in self.denoising_regs:
            print("### Info: adding moco params regressors ...")
            txtyn = "1"
            # edit moco files in txt
            if not os.path.exists(os.path.join(sps, f"Nuisance/{moco}_nohdr.txt")):
                run_string = f"cd {sps}; tail -n +2 Moco/{moco_params_name}.tsv > Nuisance/{moco}_nohdr.txt"
                print(run_string)
                os.system(run_string)

            add_outname += "_moco"

            add_moco = f"Nuisance/{moco}_nohdr.txt"

        if "compcor" in self.denoising_regs:
            txtyn = "1"
            print("### Info: adding compcor regressors ...")
            # edit compcor txt file
            compcorname = "regressors_compcor"
            if len(fmriname.split('_')) != 1:
                compcorname += "_st"

            self._compcor_persub(sps)
            # remove header from compcor file
            if not os.path.exists(os.path.join(sps, f'Nuisance/{compcorname}_nohdr.txt')):
                run_string_cc = f"cd {sps}; tail -n +2 {compcorname}.txt > Nuisance/{compcorname}_nohdr.txt"
                print(run_string_cc)
                os.system(run_string_cc)

            add_outname += "_cc"
            add_compcor = f"Nuisance/{compcorname}_nohdr.txt"

        # Combination of denoising regressors
        if len(add_compcor)!= 0 or len(add_moco)!= 0 or len(add_outliers) != 0:
            run_string = f"cd {sps}; paste -d '\t' {add_outliers} {add_compcor} {add_moco} > Nuisance/{nuisance_txt}.txt"
            print(run_string)
            os.system(run_string)
        
        if len(add_compcor)==0 and len(add_moco)==0 and len(add_outliers)==0:
            nuispath = ''
        else:
            nuispath = os.path.join(sps,f"Nuisance/{nuisance_txt}.txt")

        ## end voxel-wise

        if len(self.denoising_regs) == 5:
            # which means all combination together
            denoised_name += "_all"
        else:
            denoised_name += add_outname

        # check if the noise_regression folder exists before and delete the folder in case
        if os.path.exists(os.path.join(sps, f'{noise_reg_out}.feat')):
            os.system('rm -rf %s' % os.path.join(sps, f'{noise_reg_out}.feat'))
            print("!!! WARNING: Removing existing noise_regression folder")

       
        os.system(f'export DIREC={sps}; \
                    export FSL_TEMP={self.fsl_template_dir}; \
                    export PNMPATHS={pnmpaths};\
                    export TXTYN={txtyn};\
                    export OUTDIRNR={noise_reg_out};\
                    export FMRINAME=m{fmriname};\
                    export NUISPATH={nuispath};\
                    bash {self.working_dir}/noisereg_subtemplate_new.sh')

        if os.uname()[1] in ['miplabsrv3','miplabsrv4']:
            add_paths = 'export PATH="/usr/bin/:$PATH";'
            fsl_feat = 'fsl5.0-feat'
            tmpout = ''
        elif os.uname()[1] in ['stiitsrv21','stiitsrv22','stiitsrv23']:
            add_paths = 'FSLDIR=/usr/local/fsl; . ${FSLDIR}/etc/fslconf/fsl.sh; PATH=${FSLDIR}/bin:${PATH}; export FSLDIR PATH;'
            fsl_feat = 'feat'
            tmpout = '> /home/iricchi/tmp/out_feat.txt'

        print("### Info: running denoising...")
        run_string_final = f'{add_paths} cd {sps}; {fsl_feat} design_noiseregression.fsf {tmpout}; cp {noise_reg_out}.feat/stats/res4d.nii.gz m{fmriname}_{denoised_name}.nii.gz;\
                             fslcpgeom m{fmriname}.nii.gz m{fmriname}_{denoised_name}.nii.gz'
        print(run_string_final)
        os.system(run_string_final)

    def normalize(self):
        subj_paths = [os.path.join(s, self.func) for s in self.list_subjects]

        # # (un)comment
        # s = self.list_subjects[0]
        # print(s)
        # subj_paths = [os.path.join(s, self.func)]
        #

        print("### Info: Starting normalization ...") 

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="multiprocessing")(delayed(self._normalization)(sub, self.fmriname)\
                 for sub in subj_paths)

        print("### Info: normalization done in %.3f s" %(time.time() - start ))
    
    def _normalization(self, sps, fmriname="fmri"):
        if self.version == 0:
            add = ''
        elif self.version == 1:
            add = '_1'
        elif self.version == 2:
            add = '_2'

        if hasattr(self, 'pam50_template'):
            PAM50_TEMP = self.pam50_template
        else:
            PAM50_TEMP = os.path.join(self.SCT_PATH,'data','PAM50','template','PAM50_t2.nii.gz')

        run_string = f'cd {sps}; sct_apply_transfo -i m{fmriname}_denoised.nii.gz -d {PAM50_TEMP} -w Normalization/warp_fmri2template{add}.nii.gz\
                     -x linear -o m{fmriname}_denoised_n.nii.gz' 

        print(run_string)
        os.system(run_string)

        return

    def apply_smoothing(self):
        subj_paths = [os.path.join(s, self.func) for s in self.list_subjects]

        # # (un)comment
        # s = self.list_subjects[2]
        # print(s)
        # subj_paths = [os.path.join(s, self.func)]
        #

        print("### Info: Appllying smoothing ...") 

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="multiprocessing")(delayed(self._fslsct_smoothing)(sub, self.fmriname)\
                 for sub in subj_paths)

        print("### Info: smoothing done in %.3f s" %(time.time() - start ))

        _send_message_telegram("smoothing")
        return

    def _fslsct_smoothing(self, sps, fmriname="fmri"):

        denoised_name = "denoised"
        add_outname = ''

        if "pnm" in self.denoising_regs:
            add_outname += "_pnm"
        if "csf" in self.denoising_regs and "pnm" not in self.denoising_regs:        
            add_outname += "_csf"
        elif "csf" in self.denoising_regs and "pnm" in self.denoising_regs:
            add_outname += "_csf"
        
        if "outliers" in self.denoising_regs:
            add_outname += "_outl"
        if "moco" in self.denoising_regs:
            add_outname += "_moco"
        if "compcor" in self.denoising_regs:
            add_outname += "_cc"
        if len(self.denoising_regs) == 5:
            denoised_name += "_all"
        else:
            denoised_name += add_outname
        
        mfmri_input = "m" +fmriname+'_'+denoised_name
        print(mfmri_input)
        # if the normalization has been run before (which means that it was a population study)
        # use smoothing with sct_math on the normalized images
        # otherwise smooth the denoised images directly
        if os.path.exists(os.path.join(sps,f'{mfmri_input}_n.nii.gz')):
            # if normalization is done before
            os.system(f'export DIREC={sps}; export MASK_NAME={self.mask_fname}.nii.gz; export MFMRID={mfmri_input}; bash {self.working_dir}/smoothing_math.sh')
        else:        
            os.system(f'export DIREC={sps}; export MASK_NAME={self.mask_fname}.nii.gz; export MFMRID={mfmri_input}; bash {self.working_dir}/smoothing.sh')

    def prepare_for_ta(self): 
        start = time.time()
        subj_paths = [os.path.join(s, self.func) for s in self.list_subjects]

        # ## (un)comment
        # s = self.list_subjects[12]
        # print(s)
        # subj_paths = [os.path.join(s, self.func)]
        # ##
        
        # subs_oi = []
        # for s in self.list_subjects:
        #     if s.split('/')[-1] in ['LU_AT' ,'LU_EP' ,'LU_FB', 'LU_GL' ,'LU_GP' ,'LU_MD' ,'LU_VS']:
        #         subs_oi.append(s)
            
        # subj_paths = [os.path.join(s, self.func) for s in subs_oi]
        print(subj_paths)

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="multiprocessing")(delayed(self._prep_for_ta)(sub,self.fmriname)\
                 for sub in subj_paths)

        print("### Info: preparetion for TA done in %.3f s" %(time.time() - start ))
        _send_message_telegram("TA folder preparation!")
        return    

    def _prep_for_ta(self, sps,fmriname="fmri"):

        if os.path.exists(os.path.join(sps, 'TA')):
            print("Removing TA folder...")
            os.system('rm -rf %s' % os.path.join(sps, 'TA'))

        denoised_name = "denoised"
        add_outname = ''

        if "pnm" in self.denoising_regs:
            add_outname += "_pnm"
        if "csf" in self.denoising_regs and "pnm" not in self.denoising_regs:        
            add_outname += "_csf"
        elif "csf" in self.denoising_regs and "pnm" in self.denoising_regs:
            add_outname += "_csf"
        
        if "outliers" in self.denoising_regs:
            add_outname += "_outl"
        if "moco" in self.denoising_regs:
            add_outname += "_moco"
        if "compcor" in self.denoising_regs:
            add_outname += "_cc"
        if len(self.denoising_regs) == 5:
            denoised_name += "_all"
        else:
            denoised_name += add_outname
        
        mfmri_input = "m" +fmriname+'_'+denoised_name
        print(mfmri_input)

        os.system(f'export DIREC={sps}; export MASK_NAME={self.mask_fname}.nii.gz; export MFMRID={mfmri_input};  bash {self.working_dir}/prep_for_ta.sh')


if __name__ == '__main__':
    
    if len(sys.argv) < 3:
        print("**************************")
        print("ERROR: no option has been specified. Please specify what process to run. If this\
               was done, specify the username in the following way --userid username")
        print("**************************")
        assert len(sys.argv) >= 3
    else:   

        if '--config' not in sys.argv:
            print("ERROR: specify option --config with the corresponding configuration file")
            assert '--config' in sys.argv
        else:
            ind = sys.argv.index('--config') + 1
            config_file = sys.argv[ind]
            print(" ### Info: reading config file: %s" % config_file)
    
    print(" ### Info: running Preprocessing ... ")  

    PR = PreprocessingRS(config_file)

    if '--anat_norm' in sys.argv:
        print(" ### Info: running anatomic normalization to template ...")
        PR.anat_norm = True 
    if '--slice_timing' in sys.argv:
        PR.slice_timing = True
    if '--moco' in sys.argv:
        print(" ### Info: running motor correction ...")
        PR.moco = True 
    if '--func_norm' in sys.argv:
        PR.func_norm = True
    if '--pnm0' in sys.argv:
        PR.pnm0_preptxt = True
    if '--pnm1' in sys.argv:
        PR.pnm1_stage1 = True
    if '--pnm2' in sys.argv:
        # check peaks
        ind = sys.argv.index('--pnm2') + 1
        # it will specified the mode: auto or subject 'name'
        PR.mode = sys.argv[ind] 
        PR.pnm2_peaks = True
    if '--pnm3' in sys.argv:
        # generate evs
        # specify whether to read the automatically detetected peaks or not
        ind = sys.argv.index('--pnm3') + 1
        PR.mode = sys.argv[ind] 
        PR.pnm3_gen_evs = True
    if '--denoising' in sys.argv:
        PR.denoising = True
    if '--normalization' in sys.argv:
        PR.normalization = True
    if '--smoothing' in sys.argv:
        PR.smoothing = True
    if '--prep_for_ta' in sys.argv:
        PR.prep_for_ta = True

    # use this entry if you want to use a specific normalization
    if '--version' in sys.argv:
        v_ind = sys.argv.index('--version') + 1
        PR.version = sys.argv[v_ind] 

    PR.processes()





