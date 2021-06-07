# 3D Point cloud completion
This is the PyTorch implementation of PCN and TopNet. PCN and TopNet is an autoencoder for point cloud completion. As for the details of the paper, please refer to [PCN](https://arxiv.org/abs/1808.00671) and [TopNet](https://ieeexplore.ieee.org/document/8953650).


### Datasets
We use the ShapeNet and KITTI datasets in our experiments, which are available below:
* [ShapeNet](https://drive.google.com/file/d/1knz2xWiiwqR_pKa8gV8rnpf4nZkX_cnG/view?usp=sharing)
* [KITTI](https://drive.google.com/file/d/130PXvRInzfNMGh7ss2ZXF3kfwh7oqHOQ/view?usp=sharing)


### Pretrained Models
The pretrained models on ShapeNet are available as follows:
* [PCN for ShapeNet](https://drive.google.com/drive/folders/1-RjCiX1OJ0yc8p4LC26xm7EO0TIVLeO-?usp=sharing)
* [TopNet for ShapeNet](https://drive.google.com/drive/folders/1CM-NSYOAmLnTt9sjkVg057GKvxSozkeL?usp=sharing)


### Prerequisites
* Python 3.6.13
* CUDA 11.1
* Pytorch 1.6.0+cu101
* Open3D ```python -m pip install open3d==10.0```
* transform3d ```pip install transfrom3d```
* h5py ```pip install h5py```

\hr

### Usage
#### Download dataset
We use two datasets in our project.
  1. ShapeNet
    Download it from the [link](https://drive.google.com/file/d/1knz2xWiiwqR_pKa8gV8rnpf4nZkX_cnG/view?usp=sharing) and move it into the folder for storing the dataset. (e.g., ```./{project_path}/shapenet```).
  2. KITTI
    Download it from the [link](https://drive.google.com/file/d/130PXvRInzfNMGh7ss2ZXF3kfwh7oqHOQ/view?usp=sharing) and move it into the folder for storing the dataset. (e.g., ```./{project_path}/kitti```).

#### Training
All log files in the training process, such as log message, checipoints, loss curve image, configuration.json, etc, will be saved to the work directory.
* Train PCN (2 GPUs)
``` python main.py --gpu_id 0,1 --save_path /path/to/logfiles/ 
                   --data_path /datapath --model pcn --npts 16384 
                   --corase 1024 --alpha 0.5 --embedding_dim 1024 
                   --batch_size 32 --optim adagra --lr 0.1e-2 --epochs 200 
                   --scaling None --rotation False --mirror_prob None 
                   --crop_prob None --mixup_prob None --emd False ```
                   
 * Train TopNet (Single GPU)
``` python main.py --gpu_id 0 --save_path /path/to/logfiles/ 
                   --data_path /datapath --model topnet --npts 16384 
                   --embedding_dim 1024 --batch_size 32 --optim adagra 
                   --lr 0.1e-2 --epochs 200 --scaling None 
                   --rotation False --mirror_prob None --crop_prob None 
                   --mixup_prob None --emd False ```
 
