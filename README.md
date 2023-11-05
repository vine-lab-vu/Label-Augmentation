# Label Augmentation Method for Medical Landmark Detection
[Label augmentation method for medical landmark detection in hip radiograph images](https://arxiv.org/abs/2309.16066)   
[Yehyun Suh](https://scholar.google.com/citations?user=5GxHvrcAAAAJ&hl=en), [Peter Chan](https://scholar.google.com/citations?user=iecV098AAAAJ&hl=en), [J. Ryan Martin](https://www.researchgate.net/profile/J-Martin-10), and [Daniel Moyer](https://scholar.google.com/citations?user=sKmoxSMAAAAJ&hl=en).


## Data Collection
- Transferring DICOM format into PNG format: https://github.com/yehyunsuh/DICOM
- Annotating landmarks: https://github.com/yehyunsuh/Landmark-Annotator

## Environment
- Ubuntu 22.04
- CUDA 11.7
- PyTorch 1.13.0

## Training 
- Environment Setup
```
conda create -n label_aug python=3.10 -y
conda activate label_aug
```
If you do not have conda downloaded in your setup, please refer to [conda installation page](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
- Clone this repository and set up directories
```
git clone https://github.com/vine-lab-vu/Label-Augmentation.git
cd Label-Augmentation
mkdir data && cd data
mkdir -p image/all txt && cd ..
```
- Put your data in the directories
```
Label-Augmentation
├─ data
│   ├─ image
│   │   ├─ all
│   │   │   ├─ 1.png
│   │   │   ├─ 2.png
│   │   │   ├─ ...
│   │   │   └─ < here goes all the images >
│   └─ txt
│       ├─ test.txt
│       └─ train.txt
├─ utility
│   ├─ dataset.py
│   ├─ log.py
│   ├─ main.py
│   ├─ model.py
│   ├─ preprocess.py
│   ├─ train.py
│   └─ visualization.py
├─ dataset.py
├─ main.py
├─ test.py
└─ train.py
```
train.txt and test.txt come from [Landmark-Annotator](https://github.com/yehyunsuh/Landmark-Annotator)
- Download libraries
```
pip3 install -r requirements.txt
```
- Start training
```
python3 main.py --dilate number_of_dilation --dilation_decrease number_of_decrease_in_dilation --dilation_epoch how_many_epochs_per_each_dilation --image_resize size_of_resized_image --batch_size size_of_each_batch --output_channel number_of_labels 
```
If it is your first time training or have added new data, add `--preprocess` at the end of the command

## Test
```
python3 main.py --test --output_channel number_of_labels 
```
If you have changed any other arguments that is related to the model, you have to add it to the test command.

## Results
### Landmark Prediction
<img src="https://github.com/vine-lab-vu/Label-Augmentation/assets/73840274/248bdfc7-11ec-4976-ae5a-f174e4d30bd3" width="250" height="250">
<img src="https://github.com/vine-lab-vu/Label-Augmentation/assets/73840274/6d13db28-ed63-43da-b8a2-246eea9b4704" width="250" height="250">
<img src="https://github.com/vine-lab-vu/Label-Augmentation/assets/73840274/b9dbb31e-f1a2-48aa-8223-df2ab0af11b2" width="250" height="250">
<img src="https://github.com/vine-lab-vu/Label-Augmentation/assets/73840274/9b8f9275-18a8-4e66-b99f-ce344506edc1" width="250" height="250">
<img src="https://github.com/vine-lab-vu/Label-Augmentation/assets/73840274/52f08774-b11c-4bbc-ab18-1a9e9186b991" width="250" height="250">
<img src="https://github.com/vine-lab-vu/Label-Augmentation/assets/73840274/90a7f3cd-3e83-4c83-a242-f2f825b8bf06" width="250" height="250">   

### Application
<img src="https://github.com/vine-lab-vu/Label-Augmentation/assets/73840274/ffe766b6-3cf0-4a00-acec-5b79cf4ecbb8" width="250" height="250">
<img src="https://github.com/vine-lab-vu/Label-Augmentation/assets/73840274/fb2b0612-ab69-4948-a913-ea3c2022a256" width="250" height="250">
<img src="https://github.com/vine-lab-vu/Label-Augmentation/assets/73840274/00a90d09-dcc2-4563-97df-f143d840e150" width="250" height="250">

From left to right, Total Knee Arthroplasty post-surgical assessment, cup position calculation, and pelvic tilt calculation.

## Acknowledgement
This repository is built using the [segmentation-models-pytorch](https://segmentation-modelspytorch.readthedocs.io/en/latest/) library.

<!-- 
## License
This project is released under the [MIT license](). Please see the LICENSE file for more information. 
-->

## Citation
<!-- 
Yehyun Suh, Aleksander Mika, J. Ryan Martin, and Daniel Moyer. [Dilation-erosion methods for radiograph annotation in total knee replacement](https://openreview.net/forum?id=bVC9bi_-t7Y). In Medical Imaging with Deep Learning, short paper track, 2023. 
```
@inproceedings{
suh2023dilationerosion,
title={Dilation-Erosion Methods for Radiograph Annotation in Total Knee Replacement},
author={Yehyun Suh and Aleksander Mika and J. Ryan Martin and Daniel Moyer},
booktitle={Medical Imaging with Deep Learning, short paper track},
year={2023},
url={https://openreview.net/forum?id=bVC9bi_-t7Y}
}
```
-->
Yehyun Suh, Peter Chan, J. Ryan Martin, and Daniel Moyer. [Label augmentation method for medical landmark detection in hip radiograph images](https://arxiv.org/abs/2309.16066), 2023.
```
@misc{
suh2023label,
title={Label Augmentation Method for Medical Landmark Detection in Hip Radiograph Images}, 
author={Yehyun Suh and Peter Chan and J. Ryan Martin and Daniel Moyer},
year={2023},
eprint={2309.16066},
archivePrefix={arXiv},
primaryClass={cs.LG}
}
```
