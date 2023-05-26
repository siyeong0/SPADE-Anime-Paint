# Sementic Segment To Anime Illust
![Anime SPADE Drawer 2023-05-14 13-50-40](https://github.com/siyeong0/SPADE-Anime-Paint/assets/117014820/25860f3b-8d0f-4907-92a7-1afd1ce69c17)
## Demo

### Run .exe
Download <a href="https://drive.google.com/file/d/1hhCZlwer5B-HZYv5uRJwSdJJvKDznVuA/view?usp=share_link">demo</a></i> and please execute "ssti/ssti.exe"

Load the segment image to the left frmae, reference image to the middle frame; basic images are given in "resources" foloder. It will automatically generate the new illust.
### Run .py
Download <a href="https://drive.google.com/file/d/1hr0ti1igKedEfObgHxwRVAat2vCo_fQh/view?usp=share_link">pretrained models</a></i> and copy "weights" folder to the root folder.
```bash
pip install -r requirements.txt

python ssti.py
```
It was tested in Anaconda environment.

## Dataset
It used <a href="https://www.kaggle.com/datasets/splcher/animefacedataset">Anime-Portrait-Dataset</a></i> as inputs.

![anime-portrait](https://github.com/siyeong0/SSTI/assets/117014820/2b3618a2-edc9-4cb5-ad8e-f0a61e43de48)

and generated the labels with <a href="https://github.com/siyeong0/Anime-Face-Segmentation">Anime-Face-Segmentation</a></i>.

![229331131-181bbe04-259f-4649-926c-c8916a5508e3](https://github.com/siyeong0/SSTI/assets/117014820/a3c18c75-4f17-4b2b-b0d1-05ea3c951701)

## Model Config

It used <a href="https://github.com/NVlabs/SPADE">NVlabs SPADE</a></i>:.
After copy the pretrained model from "ssti/weights/" folder, you can incrementally train new model with below command.
```bash
python train.py --name ssti --dataset_mode custom --label_dir datasets/anime/train_label/ --image_dir datasets/anime/train_img/ --no_instance --label_nc 7 --niter 8 --niter_decay 8 --batchSize 4 --display_freq 10000 --save_epoch_freq 1 --use_vae --continue_train
```
It requires 8GB gpu memory.

## References
[1] <a href="https://github.com/NVlabs/SPADE">NVlabs SPADE</a></i>

[2] <a href="https://medium.com/@steinsfu/drawing-anime-face-with-simple-segmentation-mask-ca955c62ce09">Deep Learning Project — Drawing Anime Face with Simple Segmentation Mask</a></i>

[3] <a href="https://blog.paperspace.com/nvidia-gaugan-introduction/">Understanding GauGAN</a></i>
