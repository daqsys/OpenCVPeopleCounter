# OpenCV People Counter

#### Description

This is a 'simple' OpenCV based people counter using background sustraction and image processing.
Beware because this will not work with your video frame (it will not detect people well or will not detect it).  That is because the code is very tweaked. 

For better solutions please use a deep learning based SSD (Single Shot Detectors) like YOLO, Mask-RCNN and so on.

There are two files:
 - [cvPeopleCounter.py](https://github.com/issaiass/OpenCVPeopleCounter/tree/master/imgproc/cvPeopleCounter.py) - The main file
 - [cvPeopleCounter.py](https://github.com/issaiass/OpenCVPeopleCounter/tree/master/imgproc/imutils.py) - A helper file for resize and other basic functions

### Which type of software and PC I used?

* Intel Core i5-4210U CPU @ 1.7GHz 
* Windows 8.1 Pro 64-bits
* python 3.6.4
* OpenCV 3.4.1

### Usage

It expects a video input, but if no video is specified as input it uses the local camera.

```sh
$ python cvPeopleCounter.py -v <path_to_video> &
```

Below is a GIF of some frames demonstrating the usage 

![PeopleCounter](https://github.com/issaiass/OpenCVPeopleCounter/blob/master/imgs/peoplecounter.gif)

**License**

![CC BY-SA](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-sa.svg)

**Disclaimer:  Use it at your own risk!**
