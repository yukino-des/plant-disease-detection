import os
import random
import shutil


def get_batch():
    image_names = os.listdir('../VOCdevkit/VOC2007/JPEGImages')
    random.shuffle(image_names)
    old_image_names = os.listdir('../imgs')
    for old_image_name in old_image_names:
        os.remove(os.path.join('../imgs', old_image_name))
    for image_name in image_names[:10]:
        shutil.copyfile(os.path.join('../VOCdevkit/VOC2007/JPEGImages', image_name),
                        os.path.join('../imgs', image_name))


if __name__ == '__main__':
    get_batch()
