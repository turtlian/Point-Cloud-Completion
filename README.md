# 3D Point cloud completion
This is the PyTorch implementation of PCN and TopNet. PCN and TopNet is an autoencoder for point cloud completion. As for the details of the paper, please refer to [PCN](https://arxiv.org/abs/1808.00671) and [TopNet](https://ieeexplore.ieee.org/document/8953650).


## Datasets
We use the ShapeNet and KITTI datasets in our experiments, which are available below:
* [ShapeNet](https://drive.google.com/file/d/1knz2xWiiwqR_pKa8gV8rnpf4nZkX_cnG/view?usp=sharing)
* [KITTI](https://drive.google.com/file/d/130PXvRInzfNMGh7ss2ZXF3kfwh7oqHOQ/view?usp=sharing)


## Pretrained Models
The pretrained models on ShapeNet are available as follows:
* [PCN for ShapeNet](https://drive.google.com/drive/folders/1-RjCiX1OJ0yc8p4LC26xm7EO0TIVLeO-?usp=sharing)
* [TopNet for ShapeNet](https://drive.google.com/drive/folders/1CM-NSYOAmLnTt9sjkVg057GKvxSozkeL?usp=sharing)


## Prerequisites
* Python 3.6.13
* CUDA 11.1
* Pytorch 1.6.0+cu101
* Open3D ```python -m pip install open3d==10.0```
* transform3d ```pip install transfrom3d```
* h5py ```pip install h5py```

<hr/>

## Usage
### Download dataset
We use two datasets in our project.
  1. ShapeNet
    Download it from the [link](https://drive.google.com/file/d/1knz2xWiiwqR_pKa8gV8rnpf4nZkX_cnG/view?usp=sharing) and move it into the folder for storing the dataset. (e.g., ```./{project_path}/shapenet```).
  2. KITTI
    Download it from the [link](https://drive.google.com/file/d/130PXvRInzfNMGh7ss2ZXF3kfwh7oqHOQ/view?usp=sharing) and move it into the folder for storing the dataset. (e.g., ```./{project_path}/kitti```).

### Trin the model
All log files in the training process, such as log message, checipoints, loss curve image, configuration.json, etc, will be saved to the work directory.
* Train PCN (2 GPUs)
``` 
python main.py --gpu_id 0,1 --save_path /path/to/logfiles/ --data_path /datapath --model pcn --npts 16384 
               --corase 1024 --alpha 0.5 --embedding_dim 1024 --batch_size 32 --optim adagra --lr 0.1e-2 
               --epochs 200 --scaling None --rotation False --mirror_prob None --crop_prob None --mixup_prob None 
               --emd False 
```
                   
* Train TopNet (Single GPU)
``` 
python main.py --gpu_id 0 --save_path /path/to/logfiles/ --data_path /datapath --model topnet --npts 16384 
               --embedding_dim 1024 --batch_size 32 --optim adagra --lr 0.1e-2 --epochs 200 --scaling None 
               --rotation False --mirror_prob None --crop_prob None --mixup_prob None --emd False
 ```
* Output
``` 
<Your_save_path> /configuration.json
                 /<model name>_<epoch>.pth
                 /train_loss.log
                 /val_loss.log
                 /loss_curve.png
```
### Evaluate the model
* Calculate the test performance(CD, F-score) for each class of ShapeNet
``` 
python evaluate.py --gpu_id 0 --model_path /pretrained-model/path --data_path /datapath --mode test --p_class car
```

* Visualize the pre-trained model's output for each class of ShapeNet
``` 
python evaluate.py --gpu_id 0 --model_path /pretrained-model/path --data_path /datapath --mode visual --p_class car
```

* Visualize the pre-trained model's output for KITTI
``` 
python evaluate.py --gpu_id 0 --model_path /pretrained-model/path --data_path /datapath --mode visual --data kitti
```

* Output of visualization
``` 
<Your_model_path> /input_points/p_class/0.png
                                       /1.png
                                          :
                  /pred_points/p_class/0.png
                                      /1.png
                                         :
                  /target_points/p_class/0.png
                                        /1.png
                                           :
```
<hr/>

### Quantitative Results

|Chamfer * 10000|Airplane|Cabinet|Car|Chair|Lamp|Couch|Table|Watercraft|
|---------------|--------|-------|---|-----|----|-----|-----|----------|
|PCN|7.97|31.53|9.23|28.50|43.38|18.44|32.11|18.37|
|PCN+Norm|20.01|92.96|39.08|64.33|143.34|47.91|82.29|4.19|
|TopNet|5.48|21.07|7.46|21.24|32.31|14.92|22.95|13.62|
|TopNet+Norm|15.33|76.43|30.96|61.58|137.37|44.16|76.21|34.81|

### Quantitative Results


