# Selectivity-and-Competition-of-Minds-Eye

Requirements
=======

OpenPV Computational Neuroscience Toolbox

* We use the 2017 release obtained here [OpenPV 2017 release](https://github.com/PetaVision/OpenPV/releases)
* We also use MATLAB for interpreting our experimental results

Training the Model Pathways
=======

Each pathway needs to be pretrained with specific data.  In the paper we have the Face pathway and Object pathway.
Set the parameters in the input/Pretrain.lua file.  These parameters include the path to the text file that contains the path and filenames of images to be loaded, one per line.  Also set the output directory where you would like to store the data.  You can run the file using the command, 

`lua Pretrain.lua`



