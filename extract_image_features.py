import keras.preprocessing.image

from pycocotools.coco import COCO
import numpy as np
import pickle
from config import ms_coco_dir

cnn = keras.applications.xception.Xception(include_top = False, pooling='max')

annFileTrain = '{}/annotations/captions_train2017.json'.format(ms_coco_dir)
annFileVal = '{}/annotations/captions_val2017.json'.format(ms_coco_dir)
annFileTest = '{}/annotations/image_info_test2017.json'.format(ms_coco_dir)

cocoTrain = COCO(annFileTrain)
cocoVal = COCO(annFileVal)
cocoTest = COCO(annFileTest)

size_w = size_h = 299
batch_size = 1000


for coco, dataType in [
    (cocoVal, 'val2017'),
    (cocoTrain, 'train2017'),
    (cocoTest, 'test2017')]:
    imgIds = coco.getImgIds()
    num_batches = (len(imgIds) + batch_size - 1) // batch_size
    pred_dict = {}
    for i in range(0, len(imgIds), batch_size):
        print('{}/{}'.format(i//batch_size + 1, num_batches))
        imgs = []
        for imgId in imgIds[i:i+batch_size]:
            img = coco.loadImgs([imgId])[0]
            imgPath = '%s/images/%s/%s' % (ms_coco_dir, dataType, img['file_name'])
            I = keras.preprocessing.image.load_img(imgPath, target_size=(size_w, size_h))
            I = keras.preprocessing.image.img_to_array(I)
            imgs.append(I)
        imgs = np.stack(imgs)
        imgs = keras.applications.xception.preprocess_input(imgs)
        print('loaded batch')
        pred = cnn.predict(imgs)
        print('predicted batch')
        for j, imgId in enumerate(imgIds[i:i+batch_size]):
            pred_dict[imgIds[i+j]] = pred[j]
    file_name = 'xception-' + dataType + '-descriptors.pic'
    with open(file_name, 'wb') as f:
        pickle.dump(pred_dict, f)
    print('saved', file_name)

