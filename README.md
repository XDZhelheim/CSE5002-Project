# CSE5002-Project
CSE5002 Course Project by Zheng Dong

Training code is adapted from my repo [Torch-MTS](https://github.com/XDZhelheim/Torch-MTS), which I developed and maintained for my personal research.

# Environment Settings
The required packages are:
```
pytorch>=1.11
numpy
scipy
scikit-learn 
pandas
matplotlib
pyyaml
torchinfo
```

# Instructions

1. `mkdir data/` and run `scripts/gen_dataset.py`

    This will generate four `.npz` files in `data/`: `adj.npz, data.npz, label.npz, indices.npz`.

    You can skip this step because I already uploaded the four files.

2. Run models: ADFGCN, LapeGCN, MLP

    Here we use ADFGCN as an example.

    - Modify the hyper-parameters in `configs/ADFGCN.yaml`
    - Run model train-test by
        ```bash
        python scripts/train.py -m adfgcn -g <your_gpu_id>
        ```
    - Training logs will be saved in `logs/ADFGCN/`
    - Trained models will be saved in `saved_models/ADFGCN/`, with `.pt` format

    Other models are similar.

    Example training log:
    ```log
    <logs/ADFGCN-2023-05-20-20-03-08.log>

    x-(5298, 6)	y-(5298,)

    --------- ADFGCN ---------
    {
        "lr": 0.01,
        "weight_decay": 0.001,
        "milestones": [
            10
        ],
        "clip_grad": 5,
        "max_epochs": 1000,
        "early_stop": 50,
        "pass_device": true,
        "model_args": {
            "num_nodes": 5298,
            "adj_path": "../data/adj.npz",
            "input_classes_list": [
                6,
                3,
                43,
                44,
                64,
                2506
            ],
            "embedding_dim": 8,
            "output_dim": 11,
            "hidden_dim": 32,
            "node_embedding_dim": 16,
            "cheb_k": 1,
            "num_layers": 2,
            "dropout": 0.1,
            "device": "cuda:0"
        }
    }
    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    ADFGCN                                   [5298, 11]                169,536
    ├─ModuleList: 1-1                        --                        --
    │    └─Embedding: 2-1                    [5298, 8]                 48
    │    └─Embedding: 2-2                    [5298, 8]                 24
    │    └─Embedding: 2-3                    [5298, 8]                 344
    │    └─Embedding: 2-4                    [5298, 8]                 352
    │    └─Embedding: 2-5                    [5298, 8]                 512
    │    └─Embedding: 2-6                    [5298, 8]                 20,048
    ├─Linear: 1-2                            [5298, 32]                1,568
    ├─ModuleList: 1-3                        --                        --
    │    └─DFGCN: 2-7                        [5298, 32]                5,152
    │    │    └─Dropout: 3-1                 [5298, 32]                --
    │    │    └─Dropout: 3-2                 [5298, 32]                --
    │    │    └─Dropout: 3-3                 [5298, 32]                --
    │    │    └─Dropout: 3-4                 [5298, 32]                --
    │    │    └─Dropout: 3-5                 [5298, 32]                --
    │    └─DFGCN: 2-8                        [5298, 32]                5,152
    │    │    └─Dropout: 3-6                 [5298, 32]                --
    │    │    └─Dropout: 3-7                 [5298, 32]                --
    │    │    └─Dropout: 3-8                 [5298, 32]                --
    │    │    └─Dropout: 3-9                 [5298, 32]                --
    │    │    └─Dropout: 3-10                [5298, 32]                --
    ├─Sequential: 1-4                        [5298, 11]                --
    │    └─Linear: 2-9                       [5298, 32]                2,080
    │    └─ReLU: 2-10                        [5298, 32]                --
    │    └─Dropout: 2-11                     [5298, 32]                --
    │    └─Linear: 2-12                      [5298, 11]                363
    ==========================================================================================
    Total params: 205,179
    Trainable params: 205,179
    Non-trainable params: 0
    Total mult-adds (M): 134.25
    ==========================================================================================
    Input size (MB): 0.13
    Forward/backward pass size (MB): 5.21
    Params size (MB): 0.10
    Estimated Total Size (MB): 5.44
    ==========================================================================================

    Loss: CrossEntropyLoss

    2023-05-20 20:03:09.446458 Epoch 1 	Train Loss = 2.48147 Train acc = 0.02736  Val acc = 0.18200 
    2023-05-20 20:03:09.672309 Epoch 2 	Train Loss = 2.17014 Train acc = 0.20219  Val acc = 0.15600 
    2023-05-20 20:03:09.897267 Epoch 3 	Train Loss = 2.00025 Train acc = 0.18836  Val acc = 0.27000 
    2023-05-20 20:03:10.121297 Epoch 4 	Train Loss = 1.89027 Train acc = 0.25605  Val acc = 0.33800 
    2023-05-20 20:03:10.345895 Epoch 5 	Train Loss = 1.79081 Train acc = 0.30674  Val acc = 0.40000 
    
    ...
    
    2023-05-20 20:04:38.384733 Epoch 400 	Train Loss = 0.61484 Train acc = 0.75634  Val acc = 0.73200 
    2023-05-20 20:04:38.597126 Epoch 401 	Train Loss = 0.61650 Train acc = 0.75634  Val acc = 0.73200 
    2023-05-20 20:04:38.823033 Epoch 402 	Train Loss = 0.61868 Train acc = 0.75317  Val acc = 0.73800 
    2023-05-20 20:04:39.044909 Epoch 403 	Train Loss = 0.61037 Train acc = 0.76872  Val acc = 0.73200 
    2023-05-20 20:04:39.266680 Epoch 404 	Train Loss = 0.60906 Train acc = 0.76181  Val acc = 0.73400 
    Early stopping at epoch: 404
    Best at epoch 354:
    Train Loss = 0.64359
    Train acc = 0.74827 Val acc = 0.74600 
    --------- Test ---------
    Test acc = 0.70801
    Inference time: 0.02 s
    ```
