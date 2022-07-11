# SC-Preprocessing

This github folder contains 4 main folder one named "WV_common" (Working Versions) where there will be the common scripts that are approved by all members and can be run propeprly and the others contain configuration files, templates and jupyter notebooks (if any).

The initial version is the not super generalized one on resting state (v0).

The version used for lumbar RS is v1 and it contains the following steps:
1) anat/func normalization
2) Motor correction
3) PNM
4) Denoising 
5) Smoothing
6) TA (SpiCiCAPs) - matlab
7) Normalization on external scripts 

The new version for RS has a different order that can be implemented in task too (to be consistent):

1 to 4 are unaltered so the only variations are:

5) Normalization (on denoised images)
6) Smoothing
7) TA (SpiCiCAPs) - matlab / GLM (outside scripts)

## WARNING / NOTE on smoothing and normalization:
The old version and order contains the function `sct_spinalcord_smooth`, which is still used in the new version but in the subject level analysis since the normalization is generally not done, but at the population level we are using `sct_math` with the flag `-smooth` because we do not need to straigthen the spine in the template space.

(e.g. in lumbar there is no need to straighten also the cord - which is included in `sct_spinalcord_smooth`).

On normalization, the PAM50 template is by default the one saved in SCT path, if you want to use another template (e.g. lumbar cropped), there is the configuration file variable `pam50_template` to specify with the path to the template.

## WARNING / NOTE 2:
Python 3.7 > is required! Anaconda environment is suggested.

NOTE: the task pipeline needs to be tested (and maybe debugged). In particular this has not been modified with the normalization step before (like in rest). 


# Configuration file

In this section the configuration file variables are explained.

Variables ending with `path` are the necessary paths for data input and softwares usage (e.g. FSL, SCT). 
"anat", "func", "physio" correspond to the names of the folders that contain structural, functional and physiological data in each subject.

"version" is refered to **normalization to template**, generally set to 0. If set to 1 or 2, there are different types of normalizations applied (vertebral labels or cauda equina respectively).

The flag "lumbar" was delated but in order to use or not the csf mask for the functional normalization for PNM there is `csf_mask`.

"FS" is the sampling frequency. The following variables `anat_norm, moco, func_norm, pnm0, pnm1, pnm2, pnm3, denoising, normalization, smoothing, prep_for_ta` all correspond to the different steps, therefore are simple flags. `pnm_columns` is a dictionary with keys that correspond to respiratory, cardiac and trigger to indicate which column they are in the text file [note that the numbers are in matlab-style 1=first coulumn, then in the script there's a subtraction for python].

- The variable "mask_fname" has been recently added to specify the file name of the functional mask (generally by default mask_sc) but generalized to include different implementations.

The variable `mode` is still related to the PNM stage: it can be either `auto` or the name of the subject of interest. When `mode = auto`, the PNM will go through all subjects, otherwise it will be applied only on the subject of interest to check one specific output.

Lastly `pam50_template` specifies the file path of the template to use for the normalization (e.g. personalized version because cropped). If this variable is not existing in the config file, than the normal full PAM50 will be used.

**Note on the normalization flag**: if this is true, the normalization to template will be applied on subjects and the smoothing will consequently be applied using `sct_math` function (population study). If the normalization is not used (subject-wise analysis), the smoothing will be run with sct_smooth_spinalcord.




