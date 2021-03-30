# PyTorch Gaussian Keypoint Regression

### Description
* This repo provides an implementation of keypoint regression using Gaussian heatmaps in PyTorch.
  * `docker`: Contains utils for building Docker images and Docker containers to manage Python dependencies used in this project 
  * `src`: Contains model definitions, dataloaders, and visualization utils
  * `config.py`: Script for configuring hyperparameters for a training job
  * `train.py`: Script for training
  * `analysis.py`: Script for running inference on a trained model and saving predicted keypoint visualizations

* This repo provides a lightweight, general framework for supervised keypoint regression. We use it to train HULK: Hierarchical Untangling from Learned Keypoints, which predicts task-relevant keypoints for cable untangling. For more details, see:
#### ["Untangling Dense Knots by Learning Task-Relevant Keypoints"](https://sites.google.com/berkeley.edu/corl2020ropeuntangling/home)
#### Jennifer Grannen*, Priya Sundaresan*, Brijen Thananjeyan, Jeffrey Ichnowski, Ashwin Balakrishna, Minho Hwang, Vainavi Viswanath, Michael Laskey, Joseph E. Gonzalez, Ken Goldberg

### Example Renderings
<p float="left">
 <img src="https://github.com/priyasundaresan/blender-rope-sim/blob/master/images/000010_rgb.png" height="200">
 <img src="https://github.com/priyasundaresan/blender-rope-sim/blob/master/images/000015_rgb.png" height="200">
 <img src="https://github.com/priyasundaresan/blender-rope-sim/blob/master/images/000020_rgb.png" height="200">
</p>
<p float="left">
 <img src="https://github.com/priyasundaresan/blender-rope-sim/blob/master/images_depth/000010_rgb.png" height="200">
 <img src="https://github.com/priyasundaresan/blender-rope-sim/blob/master/images_depth/000015_rgb.png" height="200">
 <img src="https://github.com/priyasundaresan/blender-rope-sim/blob/master/images_depth/000020_rgb.png" height="200">
</p>
<p float="left">
 <img src="https://github.com/priyasundaresan/blender-rope-sim/blob/master/image_masks/000010_visible_mask.png" height="200">
 <img src="https://github.com/priyasundaresan/blender-rope-sim/blob/master/image_masks/000015_visible_mask.png" height="200">
 <img src="https://github.com/priyasundaresan/blender-rope-sim/blob/master/image_masks/000020_visible_mask.png" height="200">
</p>
<p float="left">
 <img src="https://github.com/priyasundaresan/blender-rope-sim/blob/master/annotated/000010_annotated.png" height="200">
 <img src="https://github.com/priyasundaresan/blender-rope-sim/blob/master/annotated/000015_annotated.png" height="200">
 <img src="https://github.com/priyasundaresan/blender-rope-sim/blob/master/annotated/000020_annotated.png" height="200">
</p>

### Getting Started/Overview
#### Repo Setup
* Clone this repo with `https://github.com/priyasundaresan/keypoints_cls.git` and switch to the `develop` branch with `git checkout --track origin/develop`
* Run `cd docker` and then `./docker_build.py`: This is a one-time step to build a Docker image called `priya-keypoints` according to the dependencies in `docker/Dockerfile`. Please note that this step may take a few minutes to run all the necessary installs.
* Check that the Docker image was built by running `docker images`, which should display a line like so:
```
priya-keypoints                      latest                          735686b1cd81        2 months ago        5.17GB
```
* Next, we will set up some directories for storing checkpoints and datasets. In the directory in which you cloned this repo, run `mkdir checkpoints`.  Then, make a folder in which you want to store datasets; it can be anywhere on your machine, but note the path to this directory.
* Configure the script `docker/docker_run.py` replacing `'/raid/priya/data/keypoints/datasets` with the path to the folder you just created
* Check that you can launch a Docker container from the image you created; `cd docker` and run `./docker_run.py` which should open a container with a prompt like:
```
root@afc66cb0930c:/host#
```
* Run `Ctrl + D` to detach out of the container
#### Dataset Generation
#### Training and Inference

### Contributing 
* If you have any features you'd like to see added or would like to contribute yourself, please let us know by contacting [Priya Sundaresan](http://priya.sundaresan.us) at priya.sundaresan@berkeley.edu or [Jennifer Grannen](http://jenngrannen.com/) at jenngrannen@berkeley.edu
