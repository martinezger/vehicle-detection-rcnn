from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

class_names = ['background', 'car', 'truck', 'bus']


class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3


# define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
# load coco model weights
rcnn.load_weights('mask_rcnn_customcar_0029.h5', by_name=True)
# load photograph
img = load_img('bus_2.png')
img = img_to_array(img)
# make prediction
results = rcnn.detect([img], verbose=0)
# get dictionary for first prediction
r = results[0]
# show photo with bounding boxes, masks, class labels and scores
print((r['rois'], r['masks'], r['class_ids'], class_names, r['scores']))
display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
