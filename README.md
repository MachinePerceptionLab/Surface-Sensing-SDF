# Surface-Sensing-SDF
## [Project page](https://machineperceptionlab.github.io/SS-SDF-page/)| [Paper]([https://arxiv.org](https://arxiv.org/pdf/2412.16467))
This is the official repo for the implementation of **Sensing Surface Patches in Volume Rendering for Inferring Signed Distance Functions**.accepted at AAAI 2025.

# Setup

## Installation
Clone the repository and create an anaconda environment called Surface-Sensing-SDF using
```
git clone git@github.com:MachinePerceptionLab/Surface-Sensing-SDF.git
cd Surface-Sensing-SDF

conda create -y -n Surface-Sensing-SDF python=3.8
conda activate Surface-Sensing-SDF

conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install cudatoolkit-dev=11.3 -c conda-forge

pip install -r requirements.txt
```
The hash encoder will be compiled on the fly when running the code.

## Dataset
For downloading the preprocessed data, run the following script. The data for the DTU, Replica, Tanks and Temples is adapted from [VolSDF](https://github.com/lioryariv/volsdf), [Nice-SLAM](https://github.com/cvg/nice-slam), and [Vis-MVSNet](https://github.com/jzhangbs/Vis-MVSNet), respectively.
```
bash scripts/download_dataset.sh
```
# Training

Run the following command to train Surface-Sensing-SDF:
```
cd ./code
CUDA_VISIBLE_DEVICES=0 python training/exp_runner.py --conf CONFIG  --scan_id SCAN_ID
```
where CONFIG is the config file in `code/confs`, and SCAN_ID is the id of the scene to reconstruct.

We provide example commands for training DTU, ScanNet, and Replica dataset as follows:
```
# DTU scan65
CUDA_VISIBLE_DEVICES=0 python training/exp_runner.py --conf confs/dtu_mlp_3views.conf  --scan_id 65

# ScanNet scan 1 (scene_0050_00)
CUDA_VISIBLE_DEVICES=0 python  training/exp_runner.py --conf confs/scannet_mlp.conf  --scan_id 1

# Replica scan 1 (room0)
CUDA_VISIBLE_DEVICES=0 python training/exp_runner.py --conf confs/replica_mlp.conf  --scan_id 1
```

We created individual config file on Tanks and Temples dataset so you don't need to set the scan_id. Run training on the courtroom scene as:
```
CUDA_VISIBLE_DEVICES=0 python training/exp_runner.py --conf confs/tnt_mlp_1.conf
```

We also generated high resolution monocular cues on the courtroom scene and it's better to train with more gpus. First download the dataset
```
bash scripts/download_highres_TNT.sh
```

# Evaluations

## DTU
First, download the ground truth DTU point clouds:
```
bash scripts/download_dtu_ground_truth.sh
```
then you can evaluate the quality of extracted meshes (take scan 65 for example):
```
python evaluate_single_scene.py --input_mesh scan65_mesh.ply --scan_id 65 --output_dir dtu_scan65
```

We also provide script for evaluating all DTU scenes:
```
python evaluate.py
```
Evaluation results will be saved to ```evaluation/DTU.csv``` by default, please check the script for more details.

## Replica
Evaluate on one scene (take scan 1 room0 for example)
```
cd replica_eval
python evaluate_single_scene.py --input_mesh replica_scan1_mesh.ply --scan_id 1 --output_dir replica_scan1
```

We also provided script for evaluating all Replica scenes:
```
cd replica_eval
python evaluate.py
```
please check the script for more details.

## ScanNet
```
cd scannet_eval
python evaluate.py
```
please check the script for more details.

## Tanks and Temples
You need to submit the reconstruction results to the [official evaluation server](https://www.tanksandtemples.org), please follow their guidance. We also provide an example of our submission [here](https://drive.google.com/file/d/1Cr-UVTaAgDk52qhVd880Dd8uF74CzpcB/view?usp=sharing) for reference.


# Citation
If you find our code or paper useful, please cite
```bibtex
@article{jiang2024sensing,
  title={Sensing Surface Patches in Volume Rendering for Inferring Signed Distance Functions},
  author={Jiang, Sijia and Wu, Tong and Hua, Jing and Han, Zhizhong},
  journal={arXiv preprint arXiv:2412.16467},
  year={2024}
}
```
