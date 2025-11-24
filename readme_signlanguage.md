# generate the keypoint

Prepare:
"
conda create -n sapiens_lite python=3.10
conda activate sapiens_lite
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python tqdm json-tricks
"

go to the lite path 


run the code:

1. multiprocess_video_processor.py edit:
"
POSE_CHECKPOINT = "/home/yunli/data/sapiens/lite/sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2"
DATASET_ROOT = "/shares/iict-sp2.ebling.cl.uzh/common/popsign_v1_0/game/"
OUTPUT_ROOT = "/home/yunli/data/sapiens/lite/game/"
"


2. run the game.sh



