#!/usr/bin/python

"""
This pipeline is used mainly for TASK.
If subjects or sessions need to be discarded, it is simply necessary to rename the folder
to discard  

Order of the options (recommended to follow):

STEPS :

N.B. for out script I'm referring to Nawal's in /home/kinany/Code/SC/CompSeq_Wrist/func

0) generate Timings
  (look at notebook(*) to generate files for GLM)
    * It contains also the motor correction parameters analysis that can be done after
    running that step (3)  

Step 0 is done outside this pipeline
From now on all steps can be done with this script:
(only exception is the generation of masks with FSLeyes)  

1a) Alignment
1b) Crop runs
1c) Crop rest

2) anat_norm (register to template, anatomical normalization)
3) moco - but here per session -> see plots generated from the notebook
   to include/exclude session/subject (see notebook(*))
   after running the motor correction it is possible to run the next step
   which is done to run the segmentation on the mean functional images

4) Prepare and average functional runs to perform segmentation on the mean only
  (using FSLeyes)
5) segmentation on the mean_for_seg by hand (out script)
6) normalization with func_norm on mean 
7) smoothing 
8) pnm run 
9) a- moco outliers creation 
Do the template fsf file to then run
10) b- feat firts flobs
11) GLM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
THIS STILL NEEDS TO BE INTEGRATE IN THE PIPELINE
12) 10_feat_second (second level - flameo) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NOTE: data should be structured in psuedo-BIDS format
    sub-001
        |_ anat
        |_ physio
            |_ sess-01 
            |_ sess-02 
        |_ func
            |_ rest 
            |_ sess-01 
            |_ sess-02 
            ...

            (the folder names can be specified in the config file with the variables
             restFolder_name and fsessions_name)
 WARNING! : the runs name should be consistent in the physio and func folders.

----
Author: Ilaria Ricchi

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
from utils import flatten_list


class PreprocessingT(object):
    """PreprocessingT performs the STEP1 for the whole Pipeline in task data.

    WARNING: for cropping images (rest and runs) they need to be placed in the specific way
    described in the NOTE above where there is the specifdic 'rest' folder and the several
    folders per session. 

    NOTE: segmentation and labelling are done separately and should be visually evaluated!
    
    Commands used:
    sct_deepseg_sc with viewer initialization
    sct_deepseg_sc -i t2.nii.gz -c t2 -centerline viewer
    sct_label_utils -i t2.nii.gz -create-viewer 2,3,4,5,6,7,8,9,10,11,12 -o labels.nii.gz
    or this to generate labels 
    sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -c t2

    For Lumbar (CSF add):
    sct_propseg -i t2.nii.gz -c t2 -o t2_seg.nii.gz -CSF
    -
    fslmaths t2_seg.nii.gz -add t2_rescaled_CSF_seg.nii.gz t2_seg_CSF.nii.gz
    sct_label_utils -i t2.nii.gz -create-viewer 18,19,20,21,22,23,24,25 -o labels.nii.gz  

    NOTES: 
        FSLeyes was used to normalize manually the functional images, by creating a mask


    """
    def __init__(self, config_file):
        super(PreprocessingT, self).__init__()
        self.config_file = config_file
        self.__init_configs()
        self.working_dir = os.getcwd()
        self.__read_configs(config_file)
        self.__save_list_runs()
        
    def __init_configs(self):
        """Initialze variables"""

        self.aligntorest = False
        self.aligntofirst = False
        self.crop_runs = False
        self.crop_params = {}

        self.rest = False   # this flag is not used for now but it could be easily
                            # set to true and implement in the pipeline the application
                            # of specific steps separately on the rest run 

        self.anat_norm = False
        self.moco = False

        self.ave_for_seg = False  # average all functional runs of task to perform segmentation

        self.func_norm = False # this is done on the mean of all sessions

        self.aligntorest = False   # alignment to rest scan 
        self.aligntofirst = False  # alignement to first scan

        self.pnm0 = False   # step 0: prepare physio rec (txt file and filter)
        self.pnm1 = False   # first step: pnm_stage 1
        self.pnm2 = False   # second step: check peaks of cardiac sig
        self.pnm3 = False   # third step: generate evs 
        self.cof = 2  # default filter parameter for pnm0
        self.mode = '' # in the case of meditation only one subject
        self.fsessions_name = ''
        self.csf_mask = False  # default / in cervical should be true
        self.denoising = False
        self.smoothing = False
        self.prep_for_ta = False

    def processes(self):

        if self.aligntofirst:
            self.align2first()
            os.chdir(self.working_dir)

        if self.aligntorest:
            self.align2rest()
            os.chdir(self.working_dir)

        if self.crop_runs:
            self.crop_runs()
            os.chdir(self.working_dir)

        if self.anat_norm:
            self.anat_seg_norm()
            os.chdir(self.working_dir)

        if self.moco:
            self.motor_correction()
            os.chdir(self.working_dir)

        if self.ave_for_seg:
            self.average_func_runs()
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
            os.chdir(self.working_dir)

        if self.smoothing:
            self.apply_smoothing()
            os.chdir(self.working_dir)

        if self.prep_for_ta:
            self.prepare_for_ta()
            os.chdir(self.working_dir)

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

    def __list_fsessions(self, sub):
        """List all the sessions per subject"""  

        list_task_sessions = glob(os.path.join(self.parent_path, sub, self.func, self.fsessions_name+'*')) 
        rest_sess = glob(os.path.join(self.parent_path, sub, self.func, self.restFoldername)) 

        return rest_sess, list_task_sessions

    def __save_list_runs(self):
        runs_list = []
        runs_rest = []
        first_run = []
        for s in self.list_subjects:
            sub = s.split('/')[-1]
            # list all sessions
            sub_runsrest, sub_runslist = self.__list_fsessions(sub)
            runs_list.extend(sub_runslist)
            runs_rest.extend([sub_runsrest]*len(sub_runslist)) # repeated for the number of runs
            first_run.extend([sub_runslist[0]]*len(sub_runslist))
            
        self.runs_list = flatten_list(runs_list)
        self.runs_rest = flatten_list(runs_rest)
        self.first_run = flatten_list(first_runs)

        assert len(runs_list) == len(runs_rest)
        assert len(runs_list) == len(first_runs)
    
    def align2rest(self):
        
        start = time.time()
        # reference is the resing runs self.runs_rest 

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._align_runs)(self.runs_list[i], self.runs_rest[i])\
                 for i in range(len(self.runs_list)))
       
        print("### Alignment runs to rest: Done!")
        print("### Info: the alignment took %.3f s" %(time.time()-start))

        return 
    
    def align2first(self):
        
        # reference is the first run
        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._align_runs)(self.runs_list[i], self.runs_rest[i])\
                 for i in range(len(self.runs_list)))
       
        print("### Alignment runs to the first: Done!")
        print("### Info: the alignment took %.3f s" %(time.time()-start))

        return

    def _align_runs(self, sps, ref):

        run_string = 'cd %s; mkdir -p Alignment; flirt -in "mfmri_mean.nii.gz" -ref %s \
        -out Alignment/rmfmri_mean.nii.gz -omat Alignment/Align2Run1.mat -bins 256 -cost leastsq -searchcost leastsq \
        -searchrx -30 30 -searchry -30 30 -searchrz -30 30 -dof 6 -refweight "Mask/mask_fmri.nii.gz" \
        -inweight Mask/mask_fmri.nii.gz -interp spline' % (sps, ref)

        print(run_string)
        os.system(run_string)

    def crop_runs(self):
        
        paths = []
        paths.extend(self.runs_list)

        if rest:
            paths.extend(list(np.unique(self.runs_rest)))

        start = time.time()
        #if self.lumbar != 0:
        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._crop_image)(sp)\
                 for sp in paths)
       
        print("### Cropping runs: Done!")
        print("### Info: the cropping took %.3f s" %(time.time()-start))

        return 

    def _crop_image(self, sp):
        
        down = self.crop_params["down"]
        up = self.crop_params["up"]
        run_string = 'cd %s; sct_crop_image -i Processing/rmfmri_notcropped.nii.gz -zmin %s -zmax %s -o rmfmri.nii.gz'\
                     % (sub, down, up)

        print(run_string)
        os.system(run_string)

    def anat_seg_norm(self):
        
        subj_paths = [os.path.join(s, self.anat) for s in self.list_subjects]

        start = time.time()
        #if self.lumbar != 0:
        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._register_to_template)(sp)\
                 for sp in subj_paths)
       
        print("### Normalization Done!")
        print("### Info: the normalization took %.3f s" %(time.time()-start))

        return

    def _register_to_template(self, sps):

        # sps = specific path subject
        run_string = 'cd %s; sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -lspinal labels.nii.gz -c t2 -param step=0,type=label,dof=Tx_Ty_Tz_Sz:step=1,type=seg,algo=centermassrot,metric=MeanSquares,iter=10,smooth=2,gradStep=0.5,slicewise=0,smoothWarpXY=2,pca_eigenratio_th=1.6:step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,iter=3,smooth=1,slicewise=0,gradStep=0.5,smoothWarpXY=2,pca_eigenratio_th=1.6:step=3,type=im,metric=CC' % sps
        
        print(run_string)
        os.system(run_string)

    def motor_correction(self):

        # subj_paths = [os.path.join(s, self.func) for s in self.list_subjects]
        paths = []
        paths.extend(self.runs_list)

        # if self.rest:
        #     paths.extend(list(np.unique(self.runs_rest)))


        start = time.time()
        print(" ### Info: checking for Mask ...")        
        
        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._create_mask)(sub)\
                 for sub in paths)


        print("### Info: Mask created in %.3f s" %(time.time() - start ))

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._moco)(sub)\
                 for sub in paths)

        print("### Info: Motor correction done in %.3f s" %(time.time() - start ))
        
        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._move_processing)(sub)\
                 for sub in paths)

        return 

    def _create_mask(self, sps):

        if not os.path.exists(os.path.join(sps, 'Mask', 'mask_fmri.nii.gz')):
            os.makedirs(os.path.join(sps, 'Mask'), exist_ok=True)

            run_string = 'cd %s; fslmaths fmri.nii.gz -Tmean fmri_mean.nii.gz; mv fmri_mean.nii.gz Mask; cd Mask; sct_get_centerline -i fmri_mean.nii.gz -c t2; sct_create_mask -i fmri_mean.nii.gz -p centerline,fmri_mean_centerline.nii.gz -size 30mm -o mask_fmri.nii.gz;' % sps
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

    def average_func_runs(self):

        paths = [os.path.join(s, self.func) for s in self.list_subjects]
        # if self.rest:
        #     paths.extend(list(np.unique(self.runs_rest)))

        start = time.time()
        print(" ### Info: averaging runs for Segmentation ...")        
        
        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._prep_for_seg)(sub)\
                 for sub in paths)
        

        print("### Info: Averaging runs done in %.3f s" %(time.time() - start ))
        
    def _prep_for_seg(self, sp):

        runs_images = glob(os.path.join(sp, self.func, self.fsessions_name+'*','Alignment','*.nii*'))
        string = ''
        for im in runs_images:
            string += im
            string += ' '

        run_string = 'cd %s; mkdir -p Segmentation; cd Segmentation; fslmerge -t rmfmri_mean_all.nii.gz %s;\
                      fslmaths rmfmri_mean_all.nii.gz -Tmean mean_for_seg.nii.gz' % (sp, string)
        print(run_string)
        os.system(run_string)

    def func_normalize(self):
        
        subj_paths = [os.path.join(s, self.func) for s in self.list_subjects]
        
        # if self.rest:
        #     paths.extend(list(np.unique(self.runs_rest)))

        print(" ### Info: Functional Normalization ...") 

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._register_multimodal)(sub)\
                 for sub in paths)

        print("### Info: Functional Normalization done in %.3f s" %(time.time() - start ))
        print(" ### Info: Concatenate trandormes fmri -> anat & anat->template ...") 

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._concat_transfo)(sub)\
                 for sub in paths)

        print("### Info: Concatenation trandormation done in %.3f s" %(time.time() - start ))
        

        return

    def _register_multimodal(self, sps):
        
        if self.lumbar:
            t2_seg = 't2_seg_CSF'
        else:
            t2_seg = 't2_seg'
        
        run_string = 'cd %s; mkdir -p Normalization; cd Normalization; sct_register_multimodal -i ../mean_for_seg.nii.gz -d ../../%s/t2.nii.gz \
               -iseg ../Segmentation/mask_sc.nii.gz -dseg ../../%s/%s.nii.gz -param step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=seg,algo=affine,metric=MeanSquares,gradStep=0.2:step=3,type=im,algo=syn,metric=CC,iter=5,shrink=2; \
               mv warp_mfmri_mean2t2.nii.gz warp_fmri2anat.nii.gz; mv warp_t22mfmri_mean.nii.gz warp_anat2fmri.nii.gz' % (sps, self.anat, self.anat, t2_seg)
        print(run_string)
        os.system(run_string)

    def _concat_transfo(self, sps):

        run_string = 'cd %s; cd Normalization; \
                      sct_concat_transfo -w warp_fmri2anat.nii.gz ../../%s/warp_anat2template.nii.gz -o warp_fmri2template.nii.gz -d %s/data/PAM50/template/PAM50_t2.nii.gz;\
                      sct_concat_transfo -w ../../%s/warp_template2anat.nii.gz warp_anat2fmri.nii.gz -o warp_template2fmri.nii.gz -d ../mean_for_seg.nii.gz'\
                      % (sps, self.anat, self.SCT_PATH, self.anat)

        run_string2 = 'cd %s; cd Normalization; sct_apply_transfo -i ../Segmentation/mean_for_seg.nii.gz -d  %s/data/PAM50/template/PAM50_t2.nii.gz -w warp_fmri2template.nii.gz \
                       -o mean_for_seg_pam50.nii.gz' % (sps, self.SCT_PATH)

        print(run_string)
        os.system(run_string)
        print(run_string2)
        os.system(run_string2)

    def prepare_physio(self):

        subj_paths = [glob(os.path.join(s, self.physio, self.fsessions_name+'*'))  for s in self.list_subjects]
        # if self.rest:
        #     paths.extend(list(np.unique(self.runs_rest)))


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
        matstructfile = glob(os.path.join(os.getcwd(), '*.mat'))

        for matfile in matstructfile:
            matstruct = loadmat(matfile)
            data = matstruct['data']

            # cof = 2 for cervical (Nawal's), in Lumbar data = 1 #
            cof = self.cof
            Wn = (cof*2)/self.FS;
            if Wn > 1.0:
                Wn = 0.99
            B,A = butter(3,Wn,'low');
            data[:,1] = filtfilt(B,A,data[:,1]);

            data = np.array(data)
            np.savetxt(matfile[:-3]+'txt', data, fmt='%.4f', delimiter='\t')

    def pnm_stage1(self):

        subj_paths = [glob(os.path.join(s, self.physio, self.fsessions_name+'*'))  for s in self.list_subjects]

        # if self.rest:
        #     paths.extend(list(np.unique(self.runs_rest)))

        print("### Info: Physiological preparetion ...") 

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._fsl_pnm_stage1)(sub)\
                 for sub in subj_paths)

        print("### Info: Physiological preparetion done in %.3f s" %(time.time() - start ))
        return

    def _fsl_pnm_stage1(self, sps):

        # fslFixText / popp
        # you can play with smoothcard param choosing 0.2-0.5
        
        run = sps.split('/')[-1]

        run_string = 'cd %s; %sbin/fslFixText ./%s.txt ./%s_input.txt; %sbin/pnm_stage1 -i ./%s_input.txt -o %s -s %s --tr=2.5 \
        --smoothcard=0.3 --smoothresp=0.1 --resp=%s --cardiac=%s --trigger=%s' % (sps, 
                                                                               self.FSL_PATH, 
                                                                               run,
                                                                               run,
                                                                               self.FSL_PATH,
                                                                               run,
                                                                               run,                                                                               
                                                                               self.FS,
                                                                               self.pnm_columns['resp'],
                                                                               self.pnm_columns['cardiac'],  
                                                                               self.pnm_columns['trigger'])                                                                               

        #/usr/local/fsl/bin/popp -i ./sub-001_input.txt -o ./Stim -s 2000 --tr=2.5 --smoothcard=0.3 --smoothresp=0.1 --resp=1 --cardiac=2 --trigger=3

        print(run_string)
        os.system(run_string)

    def pnm_stage2(self):

        subj_paths = [glob(os.path.join(s, self.physio, self.fsessions_name+'*'))  for s in self.list_subjects]
        
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
            subname = sps.split('/')[-3]
            self.single_run(sub_paths,subname)
        else:
            sub_paths = glob(os.path.join(self.parent_path, sps, self.physio, self.fsessions_name+'*'))
            subname = sps

            for sub_path in subj_paths:
                self.single_run(sub_paths,subname)

    def _single_run(self, sub_path, subname)
        run = sub_path.split('/')[-1]

        seq_input = np.loadtxt(os.path.join(sub_path, run+'_input.txt'))
        seq_card = np.loadtxt(os.path.join(sub_path, run+'_card.txt'))
        
        seq_card = (np.round(seq_card*self.FS)).astype(int)

        # Find triggers 
        col_tr = np.int(self.pnm_columns["trigger"])-1   # in python -1 indexing
        triggers = np.where(seq_input[:,col_tr]==5)[0]
        col_car = np.int(self.pnm_columns["cardiac"])-1 
        card_signal = seq_input[triggers[0]:,col_car]/10

        indices = find_peaks(card_signal)[0]

        fig = self.__interactive_plot(card_signal, indices)
        fig.write_html(os.path.join(sub_path, '%s_hr_peaks.html') % run)

        auto_detect = indices/self.FS
        np.savetxt(os.path.join(sub_path, run+'_card_auto.txt'), auto_detect, delimiter='\n', fmt='%.3f') 
        
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
        subj_paths = [glob(os.path.join(s, self.physio, self.fsessions_name+'*'))  for s in self.list_subjects]
        print("### Info: Generate EVS ...") 

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._fsl_pnm_evs)(sub)\
                 for sub in subj_paths)

        print("### Info: EVS generation done in %.3f s" %(time.time() - start ))
        return 

    def _fsl_pnm_evs(self, sub):

        subname = sub.split('/')[-3]
        if self.mode == 'auto':
            auto_mode = '_auto'
        else:
            auto_mode = ''

        if self.lumbar == False or self.csf_mask == False:
            # specify csf mask
            run_string = 'cd %s; %sbin/pnm_evs -i ../../%s/mfmri.nii.gz -c %s_card%s.txt -r Stim_resp.txt -o %s --tr=2.5 --oc=4 --or=4 --multc=2 \
            --multr=2 --csfmask="../../%s/Segmentation/mask_csf.nii.gz" --sliceorder=up --slicedir=z' % (sub, 
                                                                                                      self.FSL_PATH,
                                                                                                      self.func,
                                                                                                      subname,
                                                                                                      auto_mode,
                                                                                                      sub,
                                                                                                      self.func)
        else :
            # no csf mask
            run_string = 'cd %s; %sbin/pnm_evs -i ../../%s/mfmri.nii.gz -c %s_card%s.txt -r Stim_resp.txt -o %s --tr=2.5 --oc=4 --or=4 --multc=2 \
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
        paths = []
        paths.extend(self.runs_list)

        # if self.rest:
        #     paths.extend(list(np.unique(self.runs_rest)))

        start = time.time()    

        print("### Info: Generate motion outliers ...") 

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                verbose=100,
                backend="threading")(delayed(self._fsl_motion_outliers)(sp)\
                for sp in paths)

        print("### Info: motion outliers generation done in %.3f s" %(time.time() - start ))

        # self.notParallel_denoising(subj_paths)
        return 

    def _fsl_motion_outliers(self, sps):

        run_string = 'cd %s; cd Moco; tail -n +2 moco_params.tsv > moco_params.txt;\
        sed -e "s/\t/ /g" moco_params.txt > moco_params_space.txt; \
        cd ..; fsl_motion_outliers -i rmfmri.nii.gz -o outliers.txt —m ../Segmentation/mask_sc.nii.gz\
        -p outliers.png --dvars --nomoco; ' % (sps)
        
        print(run_string)
        if not os.path.exists(os.path.join(sps,'outliers.png')):
            os.system(run_string)

        run_string2 = 'cd %s; paste -d " " outliers.txt Moco/moco_params_space.txt > nuisance.txt' % (sps)
        run_string3 = 'cd %s; cp Moco/moco_params.txt nuisance.txt' % (sps)
        if not os.path.exists(os.path.join(sps, 'outliers.txt')): 
            print(run_string2)
            os.system(run_string2)
        else:
            print(run_string3)
            os.system(run_string3)

        # 'fslcpgeom mfmri.nii.gz Moco/moco_params_x.nii.gz; fslcpgeom mfmri.nii.gz Moco/moco_params_y.nii.gz' %(sps)
        # print(run_string2)
        # os.system(run_string2)

        # subname = sps.split('/')[-2]
        # physiopath = sps.split('/')[:-1]
        # physiopath = os.path.join('/',*physiopath, self.physio)
        # print("Prepare nuisance regressors file...")
        # run_string3 = 'cd %s; echo %s/Moco/moco_params_x.nii.gz >> %s/%s_evlist.txt; \
        #                echo %s/Moco/moco_params_y.nii.gz >> %s/%s_evlist.txt; \
        #                cp %s/%s_evlist.txt %s/regressors_evlist.txt;' % (sps, sps,
        #                                                                  physiopath,
        #                                                                  subname,
        #                                                                  sps,
        #                                                                  physiopath,
        #                                                                  subname,
        #                                                                  physiopath,
        #                                                                  subname, sps)

        
        # print(run_string3)
        # os.system(run_string3)

        # check if the noise_regression folder exists before and delete the folder in case
        if os.path.exists(os.path.join(sps, 'output_feat_first_flobs.feat')):
            os.system('rm -rf %s' % os.path.join(sps, 'output_feat_first_flobs.feat'))
            print("!!! WARNING: Removing existing noise_regression folder")

        # take run number for the timings (according to how they are saved)
        nrun = sps.split('_')[-1]
        splitted = sps.split('/')
        dir_physio = os.path.join('/', *splitted[:-2], self.physio)
        dir_func = os.path.join('/', *splitted[:-1])
        par_dirs = sps.split('/')
        par_dir = os.path.join('/', *par_dirs[:-2])

        os.system('export nrun=%s; export PAR_DIR=%s; export FSL_TEMP=%s; export DIR_PHYSIO=%s; export DIR_FUNC=%s; export DIREC=%s; \
                   bash %s/design_1strun_subtemplate.sh' \
                   % (nrun, par_dir, self.fsl_template_dir, dir_physio, dir_func, sps, self.working_dir))

        # run_string4 = 'cd %s; fsl5.0-feat ./design_noiseregression.fsf; cp noise_regression.feat/stats/res4d.nii.gz mfmri_denoised.nii.gz \
        #                fslcpgeom mfmri.nii.gz mfmri_denoised.nii.g' % (sps)

        # print(run_string4)
        # os.system(run_string4)
    
    def apply_smoothing(self):
        paths = []
        paths.extend(self.runs_list)

        # if self.rest:
        #     paths.extend(list(np.unique(self.runs_rest)))

        print("### Info: Generate motion outliers ...") 

        start = time.time()

        Parallel(n_jobs=self.n_jobs,
                 verbose=100,
                 backend="threading")(delayed(self._fsl_smoothing)(sub)\
                 for sp in paths)

        print("### Info: smoothing done in %.3f s" %(time.time() - start ))
        return

    def _fsl_smoothing(self, sps):

        os.system('export DIREC=%s; bash %s/smoothing_task.sh' % (sps, self.working_dir))

    def prepare_for_ta(self): 
        start = time.time()
        subj_paths = [os.path.join(s, self.func) for s in self.list_subjects]

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

    PR = PreprocessingT(config_file)

    if '--aligntorest' in sys.argv:
        print(" ### Info: alinging to rest scan ...")
        PR.aligntorest = True 
    if '--aligntofirst' in sys.argv:
        print(" ### Info: alinging to firts run ...")
        PR.aligntofirst = True 
    if '--cropruns' in sys.argv:
        print(" ### Info: cropping runs ...")
        PR.crop_runs = True 
    if '--croprest' in sys.argv:
        print(" ### Info: cropping rest scan ...")
        PR.crop_rest = True 
    if '--anat_norm' in sys.argv:
        # for now not used for lumbar
        print(" ### Info: running anatomic normalization to template ...")
        PR.anat_norm = True 
    if '--moco' in sys.argv:
        print(" ### Info: running motor correction ...")
        PR.moco = True 
    if '--ave_for_seg' in sys.argv:
        print(" ### Info: averaging runs for segmentation ...")
        PR.ave_for_seg = True 
    if '--func_norm' in sys.argv:
        print("### Info: running functional normalization ...")
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
        PR.pnm3 = True
    if '--denoising' in sys.argv:
        print(" ### Info: running denoising ...")
        PR.denoising = True
    if '--smoothing' in sys.argv:
        print(" ### Info: running smoothing ...")
        PR.smoothing = True
    if '--prep_for_ta' in sys.argv:
        PR.prep_for_ta = True

    PR.processes()