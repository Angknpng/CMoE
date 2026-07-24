---

# 🌟 CMoE

This repository provides the official implementation of our paper entitled **“[Taming Cascaded Mixture-of-Experts for Modality-missing Multi-modal Salient Object Detection](https://ojs.aaai.org/index.php/AAAI/article/view/37959)”** accepted by AAAI 2026.

We propose a *Cascaded Mixture-of-Experts (CMoE)* framework that effectively handles the *modality-missing challenge* in multi-modal salient object detection.  

> 📰 **News & Resources:**  
> - **[New]** Pre-trained models and predicted saliency maps are now available!

---

## 🧩 Poster

<details>
<summary><b>🖱️ Click to expand and view the AAAI 2026 Poster</b></summary>

<br>
<p align="center">
  <a href="./assets/main.pdf">
    <img src="./assets/main.png" alt="CMoE Poster" width="70%">
  </a>
</p>
<p align="center">
  <em>Click the image to download the high-resolution PDF version.</em>
</p>

</details>

---

## 📖 Citation
If you find this work useful in your research, please cite:
```bibtex
@inproceedings{wang2026taming,
  title={Taming cascaded mixture-of-experts for modality-missing multi-modal salient object detection},
  author={Wang, Kunpeng and Sun, Feifan and Chen, Keke},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={12},
  pages={9939--9947},
  year={2026}
}
```

---

## 📦 Pre-trained Models & Saliency Maps

We provide the pre-trained model weights and the predicted saliency maps (evaluated under both modality-complete and modality-missing settings) to facilitate reproducible research.

- **Baidu Pan (百度网盘):** [Download Here](https://pan.baidu.com/s/1EwJ2ps4Lg_qmxrPGNbVRKw?pwd=CMoE) (Access Code / 提取码: `CMoE`)
- **Google Drive:** *Uploading... Link will be available soon.* ⏳

---

## ⚙️ Usage

### Ⅰ. Environment Setup

1. **Install PyTorch and torchvision** (recommended via conda):

   ```bash
   conda install pytorch==1.12.0 torchvision==0.13.0 -c pytorch
   ```

2. **Install additional dependencies**:

   ```bash
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

### Ⅱ. Training Procedure

1. **Pre-train Uni-modal Experts**

     ```bash
     python -m torch.distributed.launch --nproc_per_node=2 --master_port=2024 ./CMoE-main/train_parallel_rgb.py
     python -m torch.distributed.launch --nproc_per_node=2 --master_port=2026 ./CMoE-main/train_parallel_t.py
     ```


2. **Fine-tune Multi-modal Model**

     Before starting, set the paths for the pre-trained uni-modal weights in `./CMoE-main/options.py`. Then, run:
     ```bash
     python -m torch.distributed.launch --nproc_per_node=2 --master_port=2024 ./CMoE-main/train_parallel_multi.py
     ```



### Ⅲ. Testing

To evaluate the model under both **modality-complete** and **modality-missing** conditions, follow these steps:

1. **Prepare Black Modality Inputs**:

     For each test dataset, run the following script to generate zero-value (black) images as the missing modality input:

     ```bash
     python ./CMoE-main/black.py
     ```

2. **Set Paths**:

     In `test_produce_maps.py`, configure the paths to the trained model checkpoint, test dataset folder, and the saving directory.

3. **Run Testing**:
     
     The model will automatically predict saliency results under **modality-complete** and **modality-missing** settings:
     ```bash
     python test_produce_maps.py
     ```

     
### Ⅳ. Evaluation

1. Place the **ground-truth masks** and **predicted saliency maps** into the `./Evaluation/GT/` and `./Evaluation/sal_map/` folders, respectively.
2. Open `./Evaluation/main.m` using MATLAB.
3. Specify the evaluation dataset and run the script to compute performance metrics.


---

## 🙏 Acknowledgement

The implementation of this project is based on the following link:

- [SOD Literature Tracking](https://github.com/jiwei0921/SOD-CNNs-based-code-summary-)

---

## 📬 Contact

If you have any questions, please contact us (kp.wang@foxmail.com).
