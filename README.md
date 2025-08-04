# CMoE

This repository contains the official implementation of our manuscript:

> **Taming Cascaded Mixture-of-Experts for Modality-Missing Multi-Modal Salient Object Detection**

**Note:** To comply with the anonymous review policy, external URLs are omitted. All necessary datasets and pre-trained models are publicly available and can be found via common academic repositories (e.g., GitHub or official model zoos).




## Ⅰ. Environment Setup

1. **Install PyTorch and torchvision** (recommended via conda):

   ```
   conda install pytorch==1.12.0 torchvision==0.13.0 -c pytorch
   ```

2. **Install additional dependencies**:

   ```
   pip install -r requirements.txt
   ```

3. **Download datasets**:

   - RGB-T datasets: `VT821`, `VT1000`, `VT5000`
   - RGB-D datasets: `STERE`, `SIP`, `ReDWeb-S`, `NJUD`, `NLPR`, `DUTLF-Depth`

4. **Download pre-trained backbone**:

   - Swin-B model: `swin_base_patch4_window12_384_22k.pth`

5. **Configure dataset paths**:

   - Modify `./CMoE-main/options.py` to set the paths for all datasets and models.

6. **Prepare directories** for saving logs, checkpoints, and outputs as needed.

   ​


## Ⅱ. Training Procedure

1. **Pre-train Uni-modal Experts**

     ```
     python -m torch.distributed.launch --nproc_per_node=2 --master_port=2024 ./CMoE-main/train_parallel_rgb.py
     python -m torch.distributed.launch --nproc_per_node=2 --master_port=2026 ./CMoE-main/train_parallel_t.py
     ```


2. **Fine-tune Multi-modal Model**

     Before starting, set the paths for the pre-trained uni-modal weights in `./CMoE-main/options.py`. Then, run:
     ```
     python -m torch.distributed.launch --nproc_per_node=2 --master_port=2024 ./CMoE-main/train_parallel_multi.py
     ```



## Ⅲ. Testing

To evaluate the model under both **modality-complete** and **modality-missing** conditions, follow these steps:

1. **Prepare Black Modality Inputs**:

     For each test dataset, run the following script to generate zero-value (black) images as the missing modality input:

     ```
     python ./CMoE-main/black.py
     ```

2. **Set Paths**:

     In `test_produce_maps.py`, configure the paths to the trained model checkpoint, test dataset folder, and the saving directory.

3. **Run Testing**:
     
     The model will automatically predict saliency results under **modality-complete** and **modality-missing** settings:
     ```
     python test_produce_maps.py
     ```
​


## Ⅳ. Evaluation

1. Place the **ground-truth masks** and **predicted saliency maps** into the `./Evaluation/GT/` and `./Evaluation/sal_map/` folders, respectively.
2. Open `./Evaluation/main.m` using MATLAB.
3. Specify the evaluation dataset and run the script to compute performance metrics.
