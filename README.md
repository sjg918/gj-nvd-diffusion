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

# Run
python train_net.py --num-gpus 1 --config-file configs/diffnvd.park.best.yaml --eval-only MODEL.WEIGHTS gendata/best.pth </br>

# Demo
I'm working on demo code. SORRY. </br>

# Problems
Any questions regarding my published papers and code are always welcome. </br>

# Special Thanks
[DiffusionDet: Diffusion Model for Object Detection](https://github.com/ShoufaChen/DiffusionDet) </br>
[스마트 전조등을 위한 비전 기반 야간 차량 인식](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11654189)

# Gwang Ju Dataset sharing (Koreans Only)
데이터 셋의 용량이 약 100Gb(스테레오영상 약 60,000장)가 넘어서 저의 능력으로는 온라인 공유가 어렵습니다. </br>
만약 광주 데이터 셋이 연구에 필요하시면 제가 전체 데이터 셋을 물리적인 방법(전남대학교 방문 or 출장)으로 드릴 수 있습니다. </br>
공유에 대한 요청은 언제든지 환영입니다. 감사합니다. </br>
