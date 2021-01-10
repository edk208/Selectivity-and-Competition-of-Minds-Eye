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

These are the 128x128 images used in the paper.  

[50k ImageNet images](https://www.dropbox.com/s/w4yrxkjp6qcfdsf/imagenet_128.zip?dl=0)

[10k CelebA faces](https://www.dropbox.com/s/zqh1kma45wd6rvx/CelebAfaces128_10k.zip?dl=0)


Using the Pretrained Weights
=======

If you do not want to retrain the model, you can use the pretrained weights in the pretrained_weights folder.  There are three saved weight files per pathway and there are two pretrained pathways, face and object.

Multipath Deconvolutional Competitive Algorithm (MDCA)
=======

Use the FFAImg.lua file to run the MDCA algorithm.  You will need to change the parameters in this file, including the path to a text file that contains your test images.  In addition, you will want to change the path to the 6 network weights trained above.  They are labeled as Object or Face weights in the Connections section of the lua file.  For example, look for Face/V1ToInputError_W.pvp and replace that with the trained weights created above.

`lua FFAImg.lua`

The output directory will contain the vector representations, as well as any checkpoints and temporary files generated from the inference process.

To record every step of the inference process so you can see the evolution of the reconstruction over time, use the FFAImgSlowMo.lua file.

`lua FFAImgSlowMo.lua`

Interpreting the Results
=======

The result analysis scripts are written in Matlab and found in the scripts folder.  To see the reconstructions of the dataset, use the analysisTwoMultiscale.m file.  To write out the reconstruction over time, and to see the activity of the neurons in a graph, use analysisTwoMultiscaleSlowMo.m.  To perform classification between Faces and Objects, use the classifyFace.m file.  Note that the threshold here is a parameter set to 1.4.

If for some reason Matlab throws and error, you can use the modified mlab scripts (replace the OpenPV mlab directory with this one).  It is probably because the scripts were originally written for Octave, and adapted to Matlab here.


