# MDCS 
MDCS: More Diverse Experts with Consistency Self-distillation for Long-tailed Recognition [Official, ICCV 2023]

![image](https://github.com/fistyee/MDCS/assets/8428329/32407594-9767-4a44-b26f-67be6eb07acf)

![MixPro](./fig2.png)
## 0.Citation

If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.

coming soon~

## 1.training

### (1) CIFAR100-LT 
#### Training
* run:
```
python train.py -c configs/config_cifar100_ir100_mdcs.json
```


#### Evaluate
*  run:
``` 
python test.py -r checkpoint_path
``` 



### (2) ImageNet-LT
#### Training
* run such as resnext50 400 epochs:
```
python train.py -c configs/config_imagenet_lt_resnext50_mdcs_e400.json
```

#### Evaluate
* run:
``` 
python test.py -r checkpoint_path
``` 



 

### (3) Places-LT
#### Training
* run:
```
python train_places.py -c configs/config_places_lt_resnet152_mdcs.json
```

#### Evaluate
* run:
``` 
python test_places.py -r checkpoint_path
``` 



### (4) iNaturalist 2018
#### Training
* run :
```
python train.py -c configs/config_iNaturalist_resnet50_mdcs.json
```

#### Evaluate
* run:
``` 
python test.py -r checkpoint_path
``` 
 


## 2. Requirements
* To install requirements: 
```
pip install -r requirements.txt
```
* Run in linux (may have some problems in windows)


## 3. Datasets
### (1) Four bechmark datasets 
* Please download these datasets and put them to the /data file.
* ImageNet-LT and Places-LT can be found at [here](https://drive.google.com/drive/u/1/folders/1j7Nkfe6ZhzKFXePHdsseeeGI877Xu1yf).
* iNaturalist data should be the 2018 version from [here](https://github.com/visipedia/inat_comp).
* CIFAR-100 will be downloaded automatically with the dataloader.

```
data
├── ImageNet_LT
│   ├── test
│   ├── train
│   └── val
├── CIFAR100
│   └── cifar-100-python
├── Place365
│   ├── data_256
│   ├── test_256
│   └── val_256
└── iNaturalist 
    ├── test2018
    └── train_val2018
```

### (2) Txt files
* We provide txt files for test-agnostic long-tailed recognition for ImageNet-LT, Places-LT and iNaturalist 2018. CIFAR-100 will be generated automatically with the code.
* For iNaturalist 2018, please unzip the iNaturalist_train.zip.
```
data_txt
├── ImageNet_LT
│   ├── ImageNet_LT_test.txt
│   ├── ImageNet_LT_train.txt
│   └── ImageNet_LT_val.txt
├── Places_LT_v2
│   ├── Places_LT_test.txt
│   ├── Places_LT_train.txt
│   └── Places_LT_val.txt
└── iNaturalist18
    ├── iNaturalist18_train.txt
    ├── iNaturalist18_uniform.txt
    └── iNaturalist18_val.txt 
```


## 4. Pretrained models
* For the training on Places-LT, we follow previous methods and use [the pre-trained ResNet-152 model](https://github.com/zhmiao/OpenLongTailRecognition-OLTR).
* Please download the checkpoint. Unzip and move the checkpoint files to /model/pretrained_model_places/.



## 5. Acknowledgements
The mutli-expert framework is based on [SADE](https://github.com/vanint/sade-agnosticlt) and [RIDE](). 
Strong augmentations are based on [NCL](https://github.com/Bazinga699/NCL) and [PaCo](https://github.com/dvlab-research/Parametric-Contrastive-Learning).

