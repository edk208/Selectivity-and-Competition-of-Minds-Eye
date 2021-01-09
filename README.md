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
