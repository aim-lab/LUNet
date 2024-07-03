
<h1 align="center">
  <br>
LUNet: deep learning for the segmentation of arterioles and venules in high resolution fundus images  <br>
</h1>
<p align="center">
  <a>Jonathan Fhima</a> •
  <a>Jan Van Eijgen</a> •
  <a>Marie-Isaline Billen Moulin-Romsée</a> •
  <a>Heloïse Brackenier</a>•
  <a>Hana Kulenovic</a> 

  <p align="center">
  <a>Valérie Debeuf</a> 
  <a>Marie Vangilbergen</a>•
  <a>Moti Freiman</a>•
  <a>Ingeborg Stalmans</a>•
  <a>Joachim A Behar</a>
</p>

![Alt Text](figures/lunet.png)

### Installation

First, clone this repository and run install the environment:
```bash
cd LunetV1
python -m venv lunet_env
source lunet_env/bin/activate
pip install -r requirements.txt
```

### Data preparation
Download the following dataset from the official website (https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/Z7SHGO), and organize it as follows:
   
    Lunetv1
    ├── Databases
    │   ├── UZLF
    │   │   ├── images
    │   │   ├── artery
    │   │   ├── veins

 
### Training

Run the following command:
(Original LUNet model have been trained using 8 A100-40gb GPUs).
```bash
cd LunetV1
source lunet_env/bin/activate
python -u main.py Databases/ lunet_model
```

CUDA_VISIBLE_DEVICES=2 python -u main.py ../Databases/ lunet_model


### Evaluation
Run the following command:
```bash
cd LunetV1
source lunet_env/bin/activate
python -u eval_all.py Databases/ lunet_model_best --use_TTDA True #Replace by false if you dont want to use test time data augmentation during inference
```


### Citation
If you find this code or data to be useful for your research, please consider citing the following papers.
    
    @article{fhima2024lunet,
        title={LUNet: deep learning for the segmentation of arterioles and venules in high resolution fundus images},
        author={Fhima, Jonathan and Van Eijgen, Jan and Moulin-Roms{\'e}e, Marie-Isaline Billen and Brackenier, Helo{\"\i}se and Kulenovic, Hana and Debeuf, Val{\'e}rie and Vangilbergen, Marie and Freiman, Moti and Stalmans, Ingeborg and Behar, Joachim A},
        journal={Physiological Measurement},
        volume={45},
        number={5},
        pages={055002},
        year={2024},
        publisher={IOP Publishing}
    }

    @INPROCEEDINGS{10081641,
        author={Fhima, Jonathan and Van Eijgen, Jan and Freiman, Moti and Stalmans, Ingeborg and Behar, Joachim A},
        booktitle={2022 Computing in Cardiology (CinC)}, 
        title={Lirot.ai: A Novel Platform for Crowd-Sourcing Retinal Image Segmentations}, 
        year={2022},
        volume={498},
        number={},
        pages={1-4},
        keywords={Performance evaluation;Deep learning;Image segmentation;Databases;Data science;Retina;Data models},
        doi={10.22489/CinC.2022.060}}

    @article{van2024leuven,
        title={Leuven-Haifa High-Resolution Fundus Image Dataset for Retinal Blood Vessel Segmentation and Glaucoma Diagnosis},
        author={Van Eijgen, Jan and Fhima, Jonathan and Billen Moulin-Roms{\'e}e, Marie-Isaline and Behar, Joachim A and Christinaki, Eirini and Stalmans, Ingeborg},
        journal={Scientific Data},
        volume={11},
        number={1},
        pages={257},
        year={2024},
        publisher={Nature Publishing Group UK London}}