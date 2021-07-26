# A vehicle detection Module
This Module was created for a post graduate collage course I took, It implements Mask RCNN to detect cars, trucks and buses. The 
sample video stream for this work was taken from a live cameras installed in the General Belgrano bridge this one link two province 
of Argentina, Chaco and Corrientes. [youtube channel of the live stream](https://www.youtube.com/watch?v=3FOSfwx2DEg).

## Motion Detection
For create a dataset to train and evaluate I use [motion_detection.ipynb](motion_detection.ipynb) notebook, the motion detection
developed here is base on this [blog post](https://www.analyticsvidhya.com/blog/2020/04/vehicle-detection-opencv-python/),
 what I did is instead of print conturs of vehicles as the blog did I store in a file the frames where a motion is detected.
After the video is processed I got more than one thousand pictures to use for the the Mask RCNN.

## Mask RCNN

The Mask CRNN developed in [vehicle_detection.ipynb](vehicle_detection.ipynb) is base on [Matterport](https://github.com/matterport/Mask_RCNN)
implementation and I follow the [ballon](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon) example. 
### Train
For training the model first you need create a class and extend from `mrcnn.Config.config` class and override the following variables.
```python
class CustomConfig(Config):
    NAME= ""
    IMAGES_PER_GPU= ""
    NUM_CLASSES= ""
    STEPS_PER_EPOCH= ""
    DETECTION_MIN_CONFIDENCE= ""
```
Also you need to create a class and extend from mrcnn.utils.Dataset and override the the following methods.
```python
class CustomDataset(utils.Dataset):
    def load_cars(self):
        pass
    def load_mask(self):
        pass
    def image_reference(self):
        pass
```

#### Annotations
The annotations file use by the train were created with [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/),
this implementation for the sake of simplicity only accept one class per picture.

#### Results
the results of the traning will be a `weight` file located in the `logs` folder, some similar to `mask_rcnn_NAME_XXX.h5`
this one could be relativily large araund 450MB. This one will be use for `predict`.  

### Predict
Last cell in [vehicle_detection.ipynb](vehicle_detection.ipynb) is for make prediction for a given picture. In case you 
are running in colab be sure you upload the picture to the root folder before run the cell.

## Working example
<a href="https://colab.research.google.com/github/martinezger/vehicle-detection-rcnn/blob/main/vehicle_detection.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

The above colab notebook is complete autmated and you just need to run cell by cell in order to train the model.

