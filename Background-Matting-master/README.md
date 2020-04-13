# Background Matting: The World is Your Green Screen
![alt text](https://homes.cs.washington.edu/~soumya91/paper_thumbnails/matting.png)

By Soumyadip Sengupta, Vivek Jayaram, Brian Curless, Steve Seitz, and Ira Kemelmacher-Shlizerman

This paper will be presented in IEEE CVPR 2020.

## [**Project Page**](http://grail.cs.washington.edu/projects/background-matting/)

Go to Project page for additional details and results.

## [Paper (Arxiv)](https://arxiv.org/abs/2004.00626)

## [Blog Post](https://towardsdatascience.com/background-matting-the-world-is-your-green-screen-83a3c4f0f635?source=friends_link&sk=03e1a2de548367b22139568a7c798180&gi=85b436f7c556)

## Project members ##

* [Soumyadip Sengupta](https://homes.cs.washington.edu/~soumya91/), University of Washington
* [Vivek Jayaram](http://www.vivekjayaram.com/research), University of Washington
* [Brian Curless](https://homes.cs.washington.edu/~curless/), University of Washington
* [Steve Seitz](https://homes.cs.washington.edu/~seitz/), University of Washington
* [Ira Kemelmacher-Shlizerman](https://homes.cs.washington.edu/~kemelmi/), University of Washington

### License ###
This work is licensed under the [Creative Commons Attribution NonCommercial ShareAlike 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

## Summary ##
- [Updates](#updates) 
- [Getting Started](#getting-started)
- [Inference Code on images](#run-the-inference-code-on-sample-images)
- [Inference Code on videos](#run-the-inference-code-on-sample-videos)
- [Notes on capturing images](#notes-on-capturing-images)
- [Training code (coming soon ...)](#training-code)
- [Captured Data (coming soon ...)](#dataset)
- [Citations](#citation)

## **Updates** ##
April 9, 2020
- Issues:
	- Updated alignment function in pre-processing code. Python version uses AKAZE features (SIFT and SURF is not available with opencv3), MATLAB version also provided uses SURF features.
- New features:
	- [Testing code to replace background for videos](#run-the-inference-code-on-sample-videos)

April 8, 2020
- Issues:
	- Turning off adjustExposure() for bias-gain correction in test_pre_processing.py. (Bug found, need to be fixed)
	- Incorporating 'uncropping' operation in test_background-matting_image.py. (Output will be of same resolution and aspect-ratio as input)



## Getting Started 

Clone repository: 
```
git clone https://github.com/senguptaumd/Background-Matting.git
```

Please use Python 3. Create an [Anaconda](https://www.anaconda.com/distribution/) environment and install the dependencies. Our code is tested with Pytorch=1.1.0, Tensorflow=1.14 with cuda10.0

```
conda create --name back-matting python=3.6
conda activate back-matting
```
Make sure CUDA 10.0 is your default cuda. If your CUDA 10.0 is installed in `/usr/local/cuda-10.0`, apply
```
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
export PATH=$PATH:/usr/local/cuda-10.0/bin
``` 
Install PyTorch, Tensorflow (needed for segmentation) and dependencies
```
conda install pytorch=1.1.0 torchvision cudatoolkit=10.0 -c pytorch
pip install tensorflow-gpu==1.14.0
pip install -r requirements.txt

```

Note: The code is likely to work on other PyTorch and Tensorflow versions compatible with your system CUDA. If you already have a working environment with PyTorch and Tensorflow, only install dependencies with `pip install -r requirements.txt`. If our code fails due to different versions, then you need to install specific CUDA, PyTorch and Tensorflow versions.

## Run the inference code on sample images

### Data

To perform Background Matting based green-screening, you need to capture:
- (a) Image with the subject (use `_img.png` extension)
- (b) Image of the background without the subject (use `_back.png` extension)
- (c) Target background to insert the subject (place in `data/background`)

Use `sample_data/` folder for testing and prepare your own data based on that. This data was collected with a hand-held camera.

### Pre-trained model

Please download the pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1WLDBC_Q-cA72QC8bB-Rdj53UB2vSPnXv?usp=sharing) and place `Models/` folder inside `Background-Matting/`.

Note: `syn-comp-adobe-trainset` model was trained on the training set of the Adobe dataset. This was the model used for numerical evaluation on Adobe dataset.


### Pre-processing

1. Segmentation

Background Matting needs a segmentation mask for the subject. We use tensorflow version of [Deeplabv3+](https://github.com/tensorflow/models/tree/master/research/deeplab).

```
cd Background-Matting/
git clone https://github.com/tensorflow/models.git
cd models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ../..
python test_segmentation_deeplab.py -i sample_data/input
```

You can replace Deeplabv3+ with any segmentation network of your choice. Save the segmentation results with extension `_masksDL.png`.

2. Alignment

Skip this step, if your data is captured with fixed-camera.

- For hand-held camera, we need to align the background with the input image as a part of pre-processing. We apply simple hoomography based alignment.
- We ask users to **disable the auto-focus and auto-exposure** of the camera while capturing the pair of images. This can be easily done in iPhone cameras (tap and hold for a while).

Run `python test_pre_process.py -i sample_data/input` for pre-processing. It aligns the background image `_back.png` and changes its bias-gain to match the input image `_img.png`

We used AKAZE features python code (since SURF and SIFT unavilable in opencv3) for alignment. We also provide an alternate MATLAB code (`test_pre_process.m`), which uses SURF features. MATLAB code also provides a way to visualize feature matching and alignment. Bad alignment will produce bad matting output.
Bias-gain adjustment is turned off in the Python code due to a bug, but it is present in MATLAB code. If there are significant exposure changes between the captured image and the captured background, use bias-gain adjustment to account for that.

Feel free to write your own alignment code, choose your favorite feature detector, feature matching and alignment.

### Background Matting

```bash
python test_background-matting_image.py -m real-hand-held -i sample_data/input/ -o sample_data/output/ -tb sample_data/background/0001.png
```
For images taken with fixed camera (with a tripod), choose `-m real-fixed-cam` for best results. `-m syn-comp-adobe` lets you use the model trained on synthetic-composite Adobe dataset, without real data (worse performance).

## Run the inference code on sample videos

This is almost exactly similar as that of the image with few small changes.

### Data

To perform Background Matting based green-screening, you need to capture:
- (a) Video with the subject (`teaser.mov`)
- (b) Image of the background without the subject (use `teaser_back.png` extension)
- (c) Target background to insert the subject (place in `target_back.mov`)

We provide `sample_video/` captured with hand-held camera and `sample_video_fixed/` captured with fixed camera for testing. Please [download the data](https://drive.google.com/open?id=1C_fLlL_WUP7A_ZcdKxbYVcF_T1uy1SRK) and place both folders under `Background-Matting`. Prepare your own data based on that.

### Pre-processing

1. Frame extraction:

```
cd Background-Matting/sample_video
mkdir input background
ffmpeg -i teaser.mov input/%04d_img.png -hide_banner
ffmpeg -i target_back.mov background/%04d.png -hide_banner
```

Repeat the same for `sample_video_fixed`

2. Segmentation
```
cd Background-Matting/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ../..
python test_segmentation_deeplab.py -i sample_video/input
```

Repeat the same for `sample_video_fixed`

3. Alignment

No need to run alignment for `sample_video_fixed` or videos captured with fixed-camera.

Run `python test_pre_process_video.py -i sample_video/input -v_name sample_video/teaser_back.png` for pre-processing. Alternately you can also use `test_pre_process_video.m` in MATLAB.

### Background Matting

For hand-held videos, like `sample_video`:

```bash
python test_background-matting_image.py -m real-hand-held -i sample_video/input/ -o sample_video/output/ -tb sample_video/background/
```

For fixed-camera videos, like `sample_video_fixed`:

```bash
python test_background-matting_image.py -m real-fixed-cam -i sample_video_fixed/input/ -o sample_video_fixed/output/ -tb sample_video_fixed/background/ -b sample_video_fixed/teaser_back.png
```

To obtain the video from the output frames, run:
```
cd Background-Matting/sample_video
ffmpeg -r 60 -f image2 -i output/%04d_matte.png -vcodec libx264 -crf 15 -s 1280x720 -pix_fmt yuv420p teaser_matte.mp4
ffmpeg -r 60 -f image2 -i output/%04d_compose.png -vcodec libx264 -crf 15 -s 1280x720 -pix_fmt yuv420p teaser_compose.mp4
```

Repeat same for `sample_video_fixed`

## Notes on capturing images

For best results capture images following these guidelines:
- Choose a background that is mostly static, can be both indoor and outdoor.
- Avoid casting any shadows of the subject on the background.
	- place the subject atleast few feets away from the background.
	- if possible adjust the lighting to avoid strong shadows on the background.
- Avoid large color coincidences between subject and background. (e.g. Do not wear a white shirt in front of a white wall background.)
- Lock AE/AF (Auto-exposure and Auto-focus) of the camera.
- For hand-held capture, you need to:
	- allow only small camera motion by continuing to holding the camera as the subject exists the scene.
	- avoid backgrounds that has two perpendicular planes (homography based alignment will fail) or use a background very far away.
	- The above restirctions do not apply for images captured with fixed camera (on a tripod)

	 
## Training code

We will also release the training code, which will allow users to train on labelled data and also on unlabelled real data.

**Coming soon ...**

## Dataset

We collected 50 videos with both fixed and hand-held camera in indoor and outdoor settings. We plan to release this data to encourage future research on improving background matting.

**Coming soon ...**

## Notes

We are eager to hear how our algorithm works on your images/videos. If the algorithm fails on your data, please feel free to share it with us at soumya91@cs.washington.edu. This will help us in improving our algorithm for future research. Also, feel free to share any cool results.

## Citation
If you use this code for your research, please consider citing:
```
@InProceedings{BMSengupta20,
  title={Background Matting: The World is Your Green Screen},
  author = {Soumyadip Sengupta and Vivek Jayaram and Brian Curless and Steve Seitz and Ira Kemelmacher-Shlizerman},
  booktitle={Computer Vision and Pattern Regognition (CVPR)},
  year={2020}
}
```
