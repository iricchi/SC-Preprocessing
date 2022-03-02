# SC-Preprocessing

This github folder contains 4 main folder (one for each of the members to work in parallel without risks of merging errors) and one named "WV_common" (Working Versions) where there will be the common scripts that are approved by all members and can be run propeprly.

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

1) to 4) are unaltered 
5) Normalization (on denoised images)
6) Smoothing
7) TA (SpiCiCAPs) - matlab / GLM (outside scripts)

## WARNING / NOTE on smoothing:
The old version and order contains the function `sct_spinalcord_smooth`, while the new one contains `sct_math` with the flag `-smooth` because this step in the older version was done before normalization and sct_math would not fit cause it would mix signals from different tissues types.

## WARNING / NOTE 2:
Python 3.7 > is required! Anaconda environment is suggested.

NOTE: the task pipeline needs to be tested (and maybe debugged). In particular this has not been modified with the normalization step before (like in rest). 

