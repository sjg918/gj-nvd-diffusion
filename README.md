# gj-nvd-diffusion
Implementation of  "CNN Combined With a Prior Knowledge-based Candidate Search and Diffusion Method for Nighttime Vehicle Detection"</br>
[paper link](https://link.springer.com/article/10.1007/s12555-023-0598-x)

# Weights
Google drive link gives the pre-trained weights of our network. </br>
[google drive link](https://drive.google.com/file/d/1jI8Jok-zR4QWLqzd50XbQNNhqEy4M-h5/view?usp=sharing) </br>

# Setup
First, install an older version of pytorch. (not over 2.0.0) </br>
And follow the instructions below to install Detectron2. </br>
[Detectron2 link](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) </br>
The pytorch version I tested on is 1.12.0 and the Detectron2 version is 0.6. </br>
Please modify the path of the config file and the path of the main file. </br>
I provide a small data set for testing. </br>
I have prepared 100 images and annotations in the coco data set format in the ./gendata folder. </br>
