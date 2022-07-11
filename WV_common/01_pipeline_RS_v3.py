# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
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
# import plotly.graph_objects as go
import time
from glob import glob
from joblib import Parallel, delayed
import subprocess
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks

class PreprocessingRS(object):
    """PreprocessingRS performs the STEP1 for the whole Pipeline in RS.

    NOTE: segmentation and labelling are done separately and should be visually evaluated!
    
    Commands used:
    sct_deepseg_sc with viewer initialization
    sct_deepseg_sc -i t2.nii.gz -c t2 -centerline viewer
    sct_label_utils -i t2.nii.gz -create-viewer 2,3,4,5,6,7,8,9,10,11,12 -o labels.nii.gz
    or this to generate labels for normalization 
    sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -c t2

    But here I have always labelled the spinal levels so the option of the anatomical 
    normalization and registration to template is '-lspinal'.

    For Lumbar (CSF add):
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

        self.anat_norm = False
        self.moco = False
        self.func_norm = False
        self.mask_fname = "mask_sc"
        self.pnm0 = False   # step 0: prepare physio rec (txt file and filter)
        self.pnm1 = False   # first step: pnm_stage 1
        self.pnm2 = False   # second step: check peaks of cardiac sig
        self.pnm3 = False   # third step: generate evs 
        self.cof = 2  # default filter parameter for pnm0
        self.mode = ''
        self.fsessions_name = ''
        self.csf_mask = False  # default / in cervical should be true
        self.denoising = False
        self.normalization = False
        self.smoothing = False
        self.prep_for_ta = False

    def processes(self):
        if self.anat_norm:
            self.anat_seg_norm()
            os.chdir(self.working_dir)

        if self.moco:
            self.motor_correction()
            os.chdir(self.working_dir)

        if self.func_norm:
            self.func_normalize()
            os.chdir(self.working_dir)

        if self.pnm0:
            self.prepare_physio()
            os.chdir(self.working_dir)

        if self.pnm1:
            self.pnm_stage1()
            os.chdir(self.working_dir)

        if self.pnm2:
            self.pnm_stage2()
            os.chdir(self.working_dir)

        if self.pnm3:
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
                 backend="threading")(delayed(self._register_to_template)(sub)\
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
        run_string0 = 'cd %s; sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -ldisc labels.nii.gz -c t2 \
        -param step=0,type=label,dof=Tx_Ty_Tz_Sz:step=1,type=seg,algo=centermassrot,metric=MeanSquares,iter=10,smooth=2,gradStep=0.5,slicewise=0,smoothWarpXY=2,pca_eigenratio_th=1.6:step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,iter=3,smooth=1,slicewise=0,gradStep=0.5,smoothWarpXY=2,pca_eigenratio_th=1.6:step=3,type=im,metric=CC' % sps
        
        ## no special params
        # run_string = 'cd %s; sct_register_to_template -i t2.nii.gz -s t2_seg_CSF.nii.gz -ldisc labels.nii.gz -c t2 -param step=0,type=label,dof=Tx_Ty_Tz_Sz:step=1,type=seg,algo=centermassrot,metric=MeanSquares,iter=10,smooth=2,gradStep=0.5,slicewise=0,smoothWarpXY=2,pca_eigenratio_th=1.6' % sps

        ## modifying params
        # run_string = 'cd %s; sct_register_to_template -i t2_cropped.nii.gz -s t2_seg_CSF_cropped.nii.gz -ldisc labels.nii.gz -c t2 \
        # -param step=0,type=label,dof=Tx_Ty_Tz_Sz:step=1,type=im,metric=CC' % sps

        # VERSION 1 : 2 VERTEBRAL LABELS AND FULL SEGMENTATION 
        # JC version - working!
        # run_string = 'cd %s; sct_label_utils -i labels.nii.gz -keep 18,21 -o labels_18_21.nii.gz' % sps
        run_string1 = 'cd %s; sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -ldisc labels.nii.gz -c t2 -ofolder vertebral_labels -qc qc\
                       -param step=1,type=seg,algo=centermassrot:step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,iter=3,slicewise=0:step=3,type=im,algo=syn,metric=CC,iter=3,slicewise=0'\
                       % sps

        # VERSION2: 1 SPINAL LABEL (CAUDA EQUINA) AND FULL SEGMENTATION
        # REMEBER TO Add the same label on the participant data (terminal)
        # sct_label_utils -i t2.nii.gz -create 33,239,314,99 -o label_caudaequinea.nii.gz'

        run_string2 = 'cd %s; sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -ldisc label_caudaequinea.nii.gz -c t2 -ofolder cauda_equina_label -qc qc\
                       -param step=1,type=seg,algo=centermassrot:step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,iter=3,slicewise=0:step=3,type=im,algo=syn,metric=CC,iter=3,slicewise=0'\
                       % sps
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
                 backend="threading")(delayed(self._create_mask)(sub)\
                 for sub in subj_paths)


        print("### Info: Mask created in %.3f s" %(time.time() - start ))

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._moco)(sub)\
                 for sub in subj_paths)

        print("### Info: Motor correction done in %.3f s" %(time.time() - start ))
        
        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._move_processing)(sub)\
                 for sub in subj_paths)

        return 

    def _create_mask(self, sps):

        if not os.path.exists(os.path.join(sps, 'Mask', 'mask_fmri.nii.gz')):
            os.makedirs(os.path.join(sps, 'Mask'), exist_ok=True)

            run_string = 'cd %s; fslmaths fmri.nii.gz -Tmean fmri_mean.nii.gz; mv fmri_mean.nii.gz Mask; cd Mask; sct_get_centerline -i fmri_mean.nii.gz -c t2;\
            sct_create_mask -i fmri_mean.nii.gz -p centerline,fmri_mean_centerline.nii.gz -size 30mm -o mask_fmri.nii.gz;' % sps
            print(run_string)
            os.system(run_string)
        else:
            print("### Info: the Mask is already existing!")

    def _moco(self, sps):

        run_string = 'cd %s; sct_fmri_moco -i fmri.nii.gz -m Mask/mask_fmri.nii.gz -x spline -param poly=0 -ofolder Moco; mv Moco/fmri_moco.nii.gz mfmri.nii.gz; mv Moco/fmri_moco_mean.nii.gz mfmri_mean.nii.gz' % sps
        print(run_string)
        os.system(run_string)

    def _move_processing(self, sps):
        os.makedirs(os.path.join(sps, 'Processing'), exist_ok=True)

        if os.path.exists(os.path.join(sps, 'Mask', 'mask_fmri.nii.gz')):
            run_string = 'cd %s; mv fmri.nii.gz Processing/fmri.nii.gz' % sps
            os.system(run_string)
            print("Moved fmri file in Processing.")

    def func_normalize(self):
        
        subj_paths = [os.path.join(s, self.func) for s in self.list_subjects]
        
        ## (un)comment
        # s = self.list_subjects[0]
        # print(s)
        # subj_paths = [os.path.join(s, self.func)]
        ##
        

        print(" ### Info: Functional Normalization ...") 

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._register_multimodal)(sub)\
                 for sub in subj_paths)

        print("### Info: Functional Normalization done in %.3f s" %(time.time() - start ))
        print(" ### Info: Concatenate tranformes fmri -> anat & anat->template ...") 
        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100, 
                 backend="threading")(delayed(self._concat_transfo)(sub)\
                 for sub in subj_paths)

        print("### Info: Concatenation tranformation done in %.3f s" %(time.time() - start ))
    
        return

    def _register_multimodal(self, sps):

        if self.version == 0:
            add = ''
        elif self.version == 1:
            add = '_1'
        elif self.version == 2:
            add = '_2'

        run_string = 'cd %s; mkdir -p Normalization; cd Normalization; sct_register_multimodal -i ../mfmri_mean.nii.gz -d ../../%s/t2.nii.gz \
               -iseg ../Segmentation/%s.nii.gz -dseg ../../%s/t2_seg.nii.gz -param step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=seg,algo=affine,metric=MeanSquares,gradStep=0.2:step=3,type=im,algo=syn,metric=CC,iter=5,shrink=2; \
               mv warp_mfmri_mean2t2.nii.gz warp_fmri2anat%s.nii.gz; mv warp_t22mfmri_mean.nii.gz warp_anat2fmri%s.nii.gz' % (sps, self.anat, self.mask_fname, self.anat,add,add)

        print(run_string)
        os.system(run_string)

    def _concat_transfo(self, sps):

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

        run_string = 'cd %s; cd Normalization; \
                      sct_concat_transfo -w warp_fmri2anat%s.nii.gz ../../%s/%s/warp_anat2template.nii.gz -o warp_fmri2template%s.nii.gz -d %sdata/PAM50/template/PAM50_t2.nii.gz;\
                      sct_concat_transfo -w ../../%s/%s/warp_template2anat.nii.gz warp_anat2fmri%s.nii.gz -o warp_template2fmri%s.nii.gz -d ../mfmri_mean.nii.gz'\
                      % (sps, add, self.anat, anat_warp, add, self.SCT_PATH, self.anat, anat_warp, add, add)

        print(run_string)
        os.system(run_string)

    def prepare_physio(self):

        subj_paths = [os.path.join(s, self.physio) for s in self.list_subjects]

        # (un)comment
        # s = self.list_subjects[2]
        # print(s)
        # subj_paths = [os.path.join(s, self.physio)]
        #

        print(" ### Info: Converting text file ...") 

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._convert_txt_filt)(sub)\
                 for sub in subj_paths)

        print("### Info: File conversion done in %.3f s" %(time.time() - start ))
        return

    def _convert_txt_filt(self, sub):

        os.chdir(sub)   
        matstructfile = glob(os.path.join(os.getcwd(), '*.mat'))[0]
        matstruct = loadmat(matstructfile)
        data = matstruct['data']

        cof = self.cof
        Wn = (cof*2)/self.FS;
        if Wn > 1.0:
            Wn = 0.99
        B,A = butter(3,Wn,'low');
        data[:,1] = filtfilt(B,A,data[:,1]);

        data = np.array(data)
        np.savetxt(matstructfile[:-3]+'txt', data, fmt='%.4f', delimiter='\t')

    def pnm_stage1(self):

        subj_paths = [os.path.join(s, self.physio) for s in self.list_subjects]

        # # (un)comment
        # s = self.list_subjects[2]
        # print(s)
        # subj_paths = [os.path.join(s, self.physio)]
        # #

        print("### Info: Physiological preparetion ...") 

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._fsl_pnm_stage1)(sub)\
                 for sub in subj_paths)

        print("### Info: Physiological preparetion done in %.3f s" %(time.time() - start ))
        return

    def _fsl_pnm_stage1(self, sps):

        subjname = sps.split('/')[-2]
        # fslFixText / popp

        # you can play with smoothcard param choosing 0.2-0.5

        run_string = 'cd %s; %sbin/fslFixText ./%s.txt ./%s_input.txt; %sbin/pnm_stage1 -i ./%s_input.txt -o %s -s %s --tr=2.5 \
        --smoothcard=0.3 --smoothresp=0.1 --resp=%s --cardiac=%s --trigger=%s' % (sps, 
                                                                               self.FSL_PATH, 
                                                                               subjname,
                                                                               subjname,
                                                                               self.FSL_PATH,
                                                                               subjname,
                                                                               subjname,                                                                               
                                                                               self.FS,
                                                                               self.pnm_columns['resp'],
                                                                               self.pnm_columns['cardiac'],  
                                                                               self.pnm_columns['trigger'])                                                                               

        #/usr/local/fsl/bin/popp -i ./sub-001_input.txt -o ./Stim -s 2000 --tr=2.5 --smoothcard=0.3 --smoothresp=0.1 --resp=1 --cardiac=2 --trigger=3

        print(run_string)
        os.system(run_string)

    def pnm_stage2(self):

        subj_paths = [os.path.join(s, self.physio) for s in self.list_subjects]
        
        # subj_paths = [os.path.join(s, self.physio) for s in self.list_subjects if s.split('/')[-1] == 'LU_AT']
        
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
                     backend="threading")(delayed(self._check_peaks_car_persub)(sub)\
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
        
        seq_input = np.loadtxt(os.path.join(sub_path, subname+'_input.txt'))
        seq_card = np.loadtxt(os.path.join(sub_path, subname+'_card.txt'))
        
        seq_card = (np.round(seq_card*self.FS)).astype(int)

        # Find triggers 
        col_tr = np.int(self.pnm_columns["trigger"])-1   # in python -1 indexing
        triggers = np.where(seq_input[:,col_tr]==5)[0]
        col_car = np.int(self.pnm_columns["cardiac"])-1 
        card_signal = seq_input[triggers[0]:,col_car]/10

        indices = find_peaks(card_signal)[0]

        fig = self.__interactive_plot(card_signal, indices)
        fig.write_html(os.path.join(sub_path, '%s_hr_peaks.html') % subname)

        auto_detect = indices/self.FS
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

        # # (un)comment
        # s = self.list_subjects[2]
        # print(s)
        # subj_paths = [os.path.join(s, self.physio)]
        # #

        print("### Info: Generate EVS ...") 

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._fsl_pnm_evs)(sub)\
                 for sub in subj_paths)

        print("### Info: EVS generation done in %.3f s" %(time.time() - start ))
        return 

    def _fsl_pnm_evs(self, sub):

        subname = sub.split('/')[-2]

        if self.mode == 'auto':
            auto_mode = '_auto'
        else:
            auto_mode = ''

        if self.csf_mask == True:
            # if there is a mask of the CSF only (mask_csf.nii)
            run_string = 'cd %s; %sbin/pnm_evs -i ../%s/mfmri.nii.gz -c %s_card%s.txt -r Stim_resp.txt -o %s --tr=2.5 --oc=4 --or=4 --multc=2 \
            --multr=2 --csfmask="../%s/Segmentation/mask_csf.nii.gz" --sliceorder=up --slicedir=z' % (sub, 
                                                                                                  self.FSL_PATH,
                                                                                                  self.func,
                                                                                                  subname,
                                                                                                  auto_mode,
                                                                                                  sub,
                                                                                                  self.func)
        else :
            # if no csf mask
            run_string = 'cd %s; %sbin/pnm_evs -i ../%s/mfmri.nii.gz -c %s_card%s.txt -r Stim_resp.txt -o %s/ --tr=2.5 --oc=4 --or=4 --multc=2 \
            --multr=2 --sliceorder=up --slicedir=z' % (sub, 
                                                      self.FSL_PATH,
                                                      self.func,
                                                      subname,
                                                      auto_mode,
                                                      sub)


        print(run_string)
        os.system(run_string)

        subname = sub.split('/')[-2]
        run_string2 = 'ls -1 `%s/bin/imglob -extensions %s/ev0*` > %s/%s_evlist.txt' % (self.FSL_PATH,
                                                                                       sub,
                                                                                       sub,
                                                                                       subname)

        print(run_string2)
        os.system(run_string2)


    def apply_denoising(self):
        subj_paths = [os.path.join(s, self.func) for s in self.list_subjects]
        # subj_paths = [os.path.join(s, self.func) for s in self.list_subjects if s.split('/')[-1] == 'LU_AT']
        # print(subj_paths)

        # # (un)comment
        # s = self.list_subjects[2]
        # print(s)
        # subj_paths = [os.path.join(s, self.physio)]
        #

        print("### Info: Generate motion outliers ...") 

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                verbose=100,
                backend="threading")(delayed(self._fsl_motion_outliers)(sub)\
                for sub in subj_paths)

        print("### Info: motion outliers generation done in %.3f s" %(time.time() - start ))

        return 

    def _fsl_motion_outliers(self, sps):

        run_string = 'cd %s; fsl_motion_outliers -i mfmri.nii.gz -o outliers.txt —m ../Segmentation/%s.nii.gz -p outliers.png --dvars --nomoco;' % (sps, self.mask_fname)
        print(run_string)
        os.system(run_string)
        # Copy header information from moco functional to moco parameters       
        run_string2 = 'cd %s; fslcpgeom mfmri.nii.gz Moco/moco_params_x.nii.gz; fslcpgeom mfmri.nii.gz Moco/moco_params_y.nii.gz' %(sps)
        print(run_string2)
        os.system(run_string2)

        subname = sps.split('/')[-2]
        physiopath = sps.split('/')[:-1]
        # physiopath = os.path.join('/',*physiopath, self.physio)
        phypath = os.path.join(*physiopath)
        physiopath = os.path.join('/',phypath, self.physio)
        print("Prepare nuisance regressors file...")
        run_string3 = 'cd %s; echo %s/Moco/moco_params_x.nii.gz >> %s/%s_evlist.txt;\
                       echo %s/Moco/moco_params_y.nii.gz >> %s/%s_evlist.txt; \
                       cp %s/%s_evlist.txt %s/regressors_evlist.txt;' % (sps, sps,
                                                                         physiopath,
                                                                         subname,
                                                                         sps,
                                                                         physiopath,
                                                                         subname,
                                                                         physiopath,
                                                                         subname, sps)

        
        print(run_string3)
        os.system(run_string3)

        # check if the noise_regression folder exists before and delete the folder in case
        if os.path.exists(os.path.join(sps, 'noise_regression.feat')):
            os.system('rm -rf %s' % os.path.join(sps, 'noise_regression.feat'))
            print("!!! WARNING: Removing existing noise_regression folder")

        os.system('export DIREC=%s; export FSL_TEMP=%s; bash %s/noisereg_subtemplate.sh' % (sps, self.fsl_template_dir, self.working_dir))

        run_string4 = 'export PATH="/usr/bin/:$PATH"; cd %s; fsl5.0-feat ./design_noiseregression.fsf; cp noise_regression.feat/stats/res4d.nii.gz mfmri_denoised.nii.gz;\
                       fslcpgeom mfmri.nii.gz mfmri_denoised.nii.gz' % (sps)

        print(run_string4)
        os.system(run_string4)
    

    def normalize(self, atype='after_denoise'):
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
                 backend="threading")(delayed(self._normalization)(sub)\
                 for sub in subj_paths)

        print("### Info: normalization done in %.3f s" %(time.time() - start ))
    
    def _normalization(self, sps):
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

        run_string = 'cd %s; sct_apply_transfo -i mfmri_denoised.nii.gz -d %s -w Normalization/warp_fmri2template%s.nii.gz\
                     -x linear -o mfmri_denoised_n.nii.gz' %(sps, PAM50_TEMP, add)

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

        print("### Info: Generate motion outliers ...") 

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._fslsct_smoothing)(sub)\
                 for sub in subj_paths)

        print("### Info: smoothing done in %.3f s" %(time.time() - start ))
        return

    def _fslsct_smoothing(self, sps):
        
        # if the normalization has been run before (which means that it was a population study)
        # use smoothing with sct_math on the normalized images
        # otherwise smooth the denoised images directly
        if os.path.exists(os.path.join(sps,'mfmri_denoised_n.nii.gz')):
            os.system('export DIREC=%s; bash %s/smoothing_math.sh' % (sps, self.working_dir))
        else:        
            os.system('export DIREC=%s; bash %s/smoothing.sh' % (sps, self.working_dir))

    def prepare_for_ta(self): 
        start = time.time()
        subj_paths = [os.path.join(s, self.func) for s in self.list_subjects]

        ## (un)comment
        # s = self.list_subjects[2]
        # print(s)
        # subj_paths = [os.path.join(s, self.func)]
        ##
        
        # subs_oi = []
        # for s in self.list_subjects:
        #     if s.split('/')[-1] in ['LU_AT' ,'LU_EP' ,'LU_FB', 'LU_GL' ,'LU_GP' ,'LU_MD' ,'LU_VS']:
        #         subs_oi.append(s)
            
        # subj_paths = [os.path.join(s, self.func) for s in subs_oi]
        print(subj_paths)

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._prep_for_ta)(sub)\
                 for sub in subj_paths)

        print("### Info: preparetion for TA done in %.3f s" %(time.time() - start ))

        return    

    def _prep_for_ta(self, sps):

        if os.path.exists(os.path.join(sps, 'TA')):
            print("Removing TA folder...")
            os.system('rm -rf %s' % os.path.join(sps, 'TA'))

        os.system('export DIREC=%s; bash %s/prep_for_ta.sh' % (sps, self.working_dir))


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
    if '--moco' in sys.argv:
        print(" ### Info: running motor correction ...")
        PR.moco = True 
    if '--func_norm' in sys.argv:
        PR.func_norm = True
    if '--pnm0' in sys.argv:
        PR.pnm0 = True
    if '--pnm1' in sys.argv:
        PR.pnm1 = True
    if '--pnm2' in sys.argv:
        # check peaks
        ind = sys.argv.index('--pnm2') + 1
        # it will specified the mode: auto or subject 'name'
        PR.mode = sys.argv[ind] 
        PR.pnm2 = True
    if '--pnm3' in sys.argv:
        # generate evs
        # specify whether to read the automatically detetected peaks or not
        ind = sys.argv.index('--pnm3') + 1
        PR.mode = sys.argv[ind] 
        PR.pnm3 = True
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





