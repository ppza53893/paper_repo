import os
import tensorflow as tf

def get_file(url: str, zip_name: str = None):
    if zip_name is None:
        zip_name = os.path.split(url)[1]
    image_zip = tf.keras.utils.get_file(zip_name,
                                      cache_subdir=os.path.abspath('.'),
                                      origin = url,
                                      extract = True)
    return image_zip

if __name__ == '__main__':
    get_file('http://images.cocodataset.org/zips/train2017.zip')
    get_file('http://images.cocodataset.org/zips/val2017.zip')
    get_file('http://images.cocodataset.org/zips/test2017.zip')
    try:
        get_file('http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip')
    except:
        print('Unpack failed')