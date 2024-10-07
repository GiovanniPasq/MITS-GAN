# MITS-GAN
This is the implementation of our Computers in Biology and Medicine 2024 work 'MITS-GAN: Safeguarding Medical Imaging from Tampering with Generative Adversarial Networks'. This study introduces MITS-GAN, a novel approach to prevent tampering in medical images, with a specific focus on CT scans. The approach disrupts the output of the attacker's [CT-GAN](https://github.com/ymirsky/CT-GAN) architecture by introducing imperceptible but yet precise perturbations. The original paper can be found [here](https://arxiv.org/abs/2401.09624). <br>
Please leave a star ⭐ and cite the following [paper](https://arxiv.org/pdf/2401.09624) if you use this repository for your project.<br>
```
@misc{pasqualino2024mitsgan,
      title={MITS-GAN: Safeguarding Medical Imaging from Tampering with Generative Adversarial Networks}, 
      author={Giovanni Pasqualino and Luca Guarnera and Alessandro Ortis and Sebastiano Battiato},
      year={2024},
      eprint={2401.09624},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

## MITS-GAN Architecture
<center><img src='MITS-GAN.png' width=100%/></center>

## Installation
Choose one of the two installation options:
### Google Colab
Quickstart here 👉 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/GiovanniPasq/f93f96b8d90adb29c1a2d3aab91d9abe/mits-gan.ipynb)<br>
Or load and run the ```MITS-GAN.ipynb``` on Google Colab following the instructions inside the notebook.
### Installation on your PC
Install the required dependencies by running the following command in your terminal:
```
pip install --upgrade scipy matplotlib pandas tensorflow keras SimpleITK pydicom torch
```

## Dataset
Dataset is available [here](https://github.com/ymirsky/CT-GAN#Datasets)

## Training
Create a new folder Named MITS-GAN and put inside the following files/folders:<br>
```
MITS-GAN
|
└───data/
│    └───ct_scan_1.raw
│    └───ct_scan_1.mhd
│    │        
│    |       ...
|    |
│    └───ct_scan_n.raw
│    └───ct_scan_n.mhd
|
└───models/
|     └───INJ
|     |    └───G_model_inj.h5
|     |    └───normalization.npy
|     |    └───equalization.pkl
|     |  
│     └───REM
│          └───G_model_rem.h5
|          └───normalization.npy
|          └───equalization.pkl
|
└───procedures/
│     └───attack_pipeline.py
|
└───utils/
|     └───dataloader.py
|     └───dicom_utils.py
|     └───equalizer.py
|     └───utils.py
|
│- config.py
│- discriminator.py
|- generator.py
|- scanDataset.py
|- train.py
```
Download the dataset and the pretrained model from the CT-GAN repo by contacting the authors and place them respectively inside the data and model folders. Run the following training script:
```
python3 MITS-GAN/train.py
```

## Testing
To generate Protected images set ```save_img_result=True``` inside the training script. Then use the [GUI.py](https://github.com/ymirsky/CT-GAN/blob/master/GUI.py) script provided by the authors using the protected generated images.

## Results
Qualitative results on the reconstruction task compared with images as manipulation targets.
<center><img src='qualitative_results.png' width=100%/></center>

Metric results evaluated between the following pairs on the: real-MITS-GAN, real-TAFIM, real-MITS-GAN tampered and real-TAFIM tampered. Lower values are better for RMSE and LPIPS, higher for PSNR.

| Metric        | Real (MITS-GAN) | Real ([TAFIM](https://github.com/shivangi-aneja/TAFIM)) | Tampered (MITS-GAN) | Tampered ([TAFIM](https://github.com/shivangi-aneja/TAFIM)) |
|---------------|------------------|--------------|------------------------|---------------------|
| RMSE          | 169.481          | 194.943      | 198.253                | 233.780             |
| PSNR          | 27.949           | 21.702       | 21.237                 | 21.469              |
| LPIPS         | 0.170            | 0.383        | 0.226                  | 0.391               |
