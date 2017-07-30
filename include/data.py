import pickle
import numpy as np
import os
from urllib import urlretrieve
import tarfile
import zipfile
import sys
import data_aug as data_aug

VGG_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def get_data_set(name="train", cifar=10):
    x = None
    y = None
    l = None

    maybe_download_and_extract()

    folder_name = "cifar_10" if cifar == 10 else "cifar_100"

    f = open('./data_set/' + folder_name + '/batches.meta', 'rb')
    datadict = pickle.load(f)
    f.close()
    l = datadict['label_names']

    if name is "train":
        for i in range(5):
            f = open('./data_set/' + folder_name + '/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f)
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']
            # print-(_X, type(_X))
            # _X = np.array(_X, dtype=float) / 255.0
            _X = np.array(_X)
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            cropped_data = data_aug._random_crop(_X, [32, 32, 3], padding=4)
            flipped_data = data_aug._random_flip_leftright(_X)
            _X = np.concatenate((_X, cropped_data, flipped_data), axis=0)
            _Y = np.concatenate((_Y, _Y, _Y))

            # _mu = np.mean(_X, axis=(0, 1, 2))
            # _std = np.std(_X, axis=(0, 1, 2))

            # _X = _X.reshape(-1, 32 * 32 * 3)

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)
                # mu = np.concatenate((mu, _mu), axis=0)
                # std = np.concatenate((std, _std), axis=0)
        # mu = np.mean(x, axis=(0, 1, 2))
        # std = np.std(x, axis=(0, 1, 2))
        # x = x.reshape(-1, 32 * 32 * 3)

    elif name is "test":
        f = open('./data_set/' + folder_name + '/test_batch', 'rb')
        datadict = pickle.load(f)
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        # x = np.array(x, dtype=float) / 255.0
        x = np.array(x)
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        # x = x.reshape(-1, 32 * 32 * 3)

    def dense_to_one_hot(labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

    x = x[:, :, :, ::-1]
    x = x - VGG_MEAN
    x = x.transpose([0, 3, 1, 2])

    return x, y, l


def _print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


def maybe_download_and_extract():
    main_directory = "./data_set/"
    cifar_10_directory = main_directory + "cifar_10/"
    cifar_100_directory = main_directory + "cifar_100/"
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)

        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_10 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_100 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        os.rename(main_directory + "./cifar-10-batches-py", cifar_10_directory)
        os.rename(main_directory + "./cifar-100-python", cifar_100_directory)
        os.remove(zip_cifar_10)
        os.remove(zip_cifar_100)
