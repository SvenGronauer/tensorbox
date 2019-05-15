import os
import sys
import requests
import tensorbox.common.utils as utils
import tensorflow as tf


def download_file(url, save_directory='/var/tmp/data'):
    utils.mkdir(save_directory)
    file_name = url.split('/')[-1]
    file_path = os.path.join(save_directory, file_name)

    if not os.path.isfile(file_path):
        with open(file_path, "wb") as f:
            print('Download {} into: {}'.format(file_name, save_directory))
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                    sys.stdout.flush()
    return file_path


def convert_rgb_images_to_float(image, label):
    image = tf.cast(image, tf.float32)
    # Normalize the images to [-1, 1]
    image = (image - 127.5) / 127.5

    return image, label


def type_cast(data, label, dtype=tf.float32):
    data = tf.cast(data, dtype)
    label = tf.cast(label, dtype)
    return data, label


def unzip(file):
    import tarfile
    if file.endswith("tar.gz"):
        tar = tarfile.open(file, "r:gz")
        tar.extractall()
        tar.close()
    elif file.endswith("tar"):
        tar = tarfile.open(file, "r:")
        tar.extractall()
        tar.close()
