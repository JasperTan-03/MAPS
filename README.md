# Graph RL for Semantic Segmentation

This project explores the use of graph-based reinforcement learning for semantic segmentation in both 2D images and 3D point clouds. By constructing graph representations and leveraging spatial and contextual features, the framework demonstrates adaptability across different dimensional data.  

Read the full paper [here](https://drive.google.com/file/d/14Yx7wo7U4WGIoSClWHPu0RniNntLXLSf/view?usp=drive_link).


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

To train the model, follow these steps:

1. Update the `configs/training_config.yaml` file to adjust hyperparameters and specify the graph input file path:
   - **graph_path**: Path to the input graph file (PyTorch format).
   - **gnn_hidden_dim**: Hidden dimension size for the GNN.
   - **gnn_output_dim**: Output dimension size for the GNN.
   - **dqn_hidden_dim**: Hidden dimension size for the DQN.
   - **num_classes**: Number of segmentation classes.
   - **k_hops**: Number of hops for graph aggregation.
   - **train_dir**: Directory to save training outputs and checkpoints.
   - **num_episodes**: Total number of training episodes.
   - **max_steps**: Maximum steps per episode.
   - **render**: Set to `True` to visualize training steps.

2. Start training by running the `train.py` script:
```bash
python train.py
```
