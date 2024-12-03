# MAPS
Multi-Agent Lidar 3D Point Cloud Segmentation

## Setup

1. Create an environment with the required dependencies using the provided `environment.yaml` file:
```bash
conda env create -f environment.yaml
activate maps
```
2. Download the leftImg8bit_trainvaltest (11GB) dataset from [here](https://www.cityscapes-dataset.com/downloads/) into a 'data' folder in the root directory. We specifically use the aachen datasets to train our model.
3. Setup your data directory as follows:
```
data
└── aachen_labeled
    └── gtFine_labelIds.pngs (from the cityscapes dataset)
└── aachen_raw
    └── leftImg8bit.pngs (from the cityscapes dataset)
```
4. Run the following command to preprocess the data:
```bash
python preprocess_image_data.py
```

## Training
