# PyTorch Gaussian Keypoint Regression Conditioned on Pixel

### Description
* This repo provides an implementation of keypoint regression using Gaussian heatmaps conditioned on an inputted pixel.
  * `docker`: Contains utils for building Docker images and Docker containers to manage Python dependencies used in this project 
  * `src`: Contains model definitions, dataloaders, and visualization utils
  * `config.py`: Script for configuring hyperparameters for a training job
  * `train.py`: Script for training
  * `analysis.py`: Script for running inference on a trained model and saving predicted keypoint visualizations

### Getting Started/Overview
#### Repo Setup
* Clone this repo with `https://github.com/priyasundaresan/keypoints_cls.git` and switch to the `condition_kpt` branch with `git checkout --track origin/condition_kpt`
* Run `cd docker` and then `./docker_build.py`: This is a one-time step to build a Docker image called `priya-keypoints` according to the dependencies in `docker/Dockerfile`. Please note that this step may take a few minutes to run all the necessary installs.
* Check that the Docker image was built by running `docker images`, which should display a line like so:
```
priya-keypoints                      latest                          735686b1cd81        2 months ago        5.17GB
```
* Next, we will set up some directories for storing checkpoints and datasets. In the directory in which you cloned this repo, run `mkdir checkpoints`.  Then, make a folder in which you want to store datasets; it can be anywhere on your machine, but note the path to this directory.
* Configure the script `docker/docker_run.py` replacing `/raid/jennifer` with the path to the folder you just created
* Check that you can launch a Docker container from the image you created; `cd docker` and run `./docker_run.py` which should open a container with a prompt like:
```
root@afc66cb0930c:/host#
```
* Run `Ctrl + D` to detach out of the container
#### Dataset Generation
* This is best done locally, instead of on a remote host since we use a OpenCV mouse GUI to annotate images and get paired (image, keypoint) datasets. This functionality is not tested with X11 forwarding or VirtualGL.
* We provide a sample dataset, `condition_endpoint_cable`, which contains images of knots in single cable along with its annotations for the given endpoint keypoint (either right or left), and the associated pin and pull keypoints. 
* Use the script `python annotate_real.py` which expects a folder called `images` (move `train/images` or `test/images` to the same directory level as this script). It will launch an OpenCV window where you can annotate keypoints; double click to annotate/save a point, which will be visualized as a blue circle on the image. Note that the script will simply save the keypoints you label into an npy file. The dataloader will construct the heatmaps for the conditioned keypoints to pass into the model as it is faster to do this on a GPU. You will have to update the code for the number of keypoints being saved and which keypoints are used as givens. Press `s` to skip an image and `r` to clear all annotations. The script saves the images/annotations to a folder organized as follows:
```
{folder_name}/
|-- images
|   `-- 00000.jpg
|   ...
`-- annots
    `-- 00000.npy
    ...
```
* Use the script `real_kp_augment.py` which expects a folder called `images` and `annots` (copy `{folder_name}/images` and `{folder_name}/annots` to the same directory level as this script). It will use image space and affine transformations to augment the dataset by `num_augs_per_img` and directly output the augmented images and keypoint annotations to the same `images` and `keypoints` folders
* Finally, move `images` and `annots` to the folder `train`
* Repeat the above steps on the test split
* Move the folders  `test`  and `train` to a folder with your desired dataset name
* This should produce a dataset like so:
```
<your_dataset_name>
|-- test
|   |-- images
|   `-- annots
`-- train
    |-- images
    `-- annots
```

#### Training and Inference
* Start a docker container (if not already done) with `cd docker && ./docker_run.py`
* Configure `train.py` by replacing `dataset_dir = 'nonplanar_hulk_aug_multicolor_reannot_looser_moredbl'` with `<your_dataset_name>`
* Run `python train.py`
* This will save checkpoints to `checkpoints/<your_dataset_name>`
* Update `analysis.py` by with `keypoints.load_state_dict(torch.load('checkpoints/<your_dataset_name>/model_2_1_24_<final_test_loss>.pth'))`
* Run `python analysis.py` which will save predicted heatmap keypoint predictions to `preds`

### Contributing 
* For any questions, contact [Priya Sundaresan](http://priya.sundaresan.us) at priya.sundaresan@berkeley.edu or [Jennifer Grannen](http://jenngrannen.com/) at jenngrannen@berkeley.edu
