# CMoE
This is the the source code for the paper entilted "Taming Cascaded Mixture-of-Experts for Modality-missing Multi-modal Salient Object Detection".

***Note: To comply with anonymous policies, external links are not directly provided here. The datasets and pre-trained backbone model required for this code can all be found on GitHub.***


## Usage

### Requirement

0. Download the RGB-T (`i.e., VT821, VT1000, and VT5000`) and RGB-D (`i.e., STERE,  SIP, ReDWeb-S, NJUD, NLPR, and DUTLF-Depth`) SOD datasets for training and testing as specified in our manuscript.
1. Download the pretrained parameters of the Swin-B backbone (`i.e., swin_base_patch4_window12_384_22k.pth`).
2. Create directories for the experiment and parameter files.
3. Organize dataset directories for pre-training and fine-tuning.
4. Please use `conda` to install `torch` (1.12.0) and `torchvision` (0.13.0).
5. Install other packages: `pip install -r requirements.txt`.
6. Set your path of all datasets in `./CMoE-main/options.py`.

### Pre-training

Please pre-train the uni-modal experts for RGB and Depth/Thermal modality separately using the following commands and save their corresponding weight files.

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2024 train_parallel_rgb.py
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2026 train_parallel_t.py
```

### Fine-tuning

Please set the path of the pre-trained unimodal expert in the `./CMoE-main/options.py` file and run the following command for fine-tuning.

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2024 train_parallel_multi.py
```

### Test

Please set the path of trained model, dataset, and result storage in the `test_produce_maps.py` file, and run the following command to test.

```
python test_produce_maps.py
```

### Evaluation
0. Put the ground truth and predicted results in the GT folder and sal_map folder, respectively, within the Evaluation folder.
1. Open the `./Evaluation/main.m` file in the Evaluation folder using Matlab software.
2. Set the path and the dataset to be evaluated, and run the `./Evaluation/main.m` file to obtain the results.
