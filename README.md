# MITS-GAN
This is the implementation of our submitted Artificial Intelligence in Medicine 2024 work 'MITS-GAN: Safeguarding Medical Imaging from Tampering with Generative Adversarial Networks'. This study introduces MITS-GAN, a novel approach to prevent tampering in medical images, with a specific focus on CT scans. The approach disrupts the output of the attacker's CT-GAN architecture by introducing imperceptible but yet precise perturbations. The original paper can be found [here](https://arxiv.org/pdf/2401.09624.pdf). <br>
Please leave a star ⭐ and cite the following [paper](https://arxiv.org/pdf/2401.09624.pdf) if you use this repository for your project.<br>
Models and code will be publicly available after the paper publication.
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

### Google Colab
Quickstart here 👉 [![Open In Colab]()<br>
Or load and run the ```MITS-GAN.ipynb``` on Google Colab following the instructions inside the notebook.<br>

## Dataset
Dataset is available [here](https://github.com/ymirsky/CT-GAN#Datasets)

## Training

## Testing

## Results
Qualitative results on the reconstruction task compared with images as manipulation targets.
<center><img src='qualitative_results.png' width=100%/></center>

Metric results evaluated between the following pairs on the: real-MITS-GAN, real-TAFIM, real-MITS-GAN tampered and real-TAFIM tampered. Lower values are better for RMSE and LPIPS, higher for PSNR.

| Metric        | Real (MITS-GAN) | Real (TAFIM) | Tampered (MITS-GAN T.) | Tampered (TAFIM T.) |
|---------------|------------------|--------------|------------------------|---------------------|
| RMSE          | 169.481          | 194.943      | 198.253                | 233.780             |
| PSNR          | 27.949           | 21.702       | 21.237                 | 21.469              |
| LPIPS         | 0.170            | 0.383        | 0.226                  | 0.391               |
