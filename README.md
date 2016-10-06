# Installation instructions

This project depends on the [Menpo Project](http://www.menpo.org/),
which is multi-platform (Linux, Windows, OS X) and [TensorFlow](http://tensorflow.org).
As explained in [Menpo's installation isntructions](http://www.menpo.org/installation/), it is
highly recommended to use
[conda](http://conda.pydata.org/miniconda.html) as your Python
distribution.

Once downloading and installing
[conda](http://conda.pydata.org/miniconda.html), this project can be
installed by:

**Step 1:** Create a new conda environment and activate it:
```console
$ conda create -n ibugnet python=3.5
$ source activate ibugnet
``` 

**Step 2:** Install [menpo](http://www.menpo.org/menpo/) and
[menpowidgets](http://www.menpo.org/menpowidgets/) and opencv from the menpo
channel: 
```console
(ibugnet)$ conda install -c menpo menpo menpowidgets opencv
```

**Step 3:** Install the [TensorFlow](https://www.tensorflow.org/)
dependencies following the [installation
instructions](https://www.tensorflow.org/versions/r0.10/get_started/index.html).

We have build a wheel file for the lab machines which is
```console
(ibugnet) pip install --ignore-installed --upgrade /vol/atlas/homes/gt108/pretrained_models/tensorflow-0.11.0rc0-py3-none-any.whl
```

# Training

(TODO)

# Pretrained models
## Keypoint estimation

/vol/atlas/homes/gt108/pretrained_models/keypoints/69_classes
