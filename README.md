# PyTorch Transfer Learning of VGG19 on Cifar-10 Dataset
Transfer Learning of VGG19 trained using ImageNet an retrained for Cifar-10 Dataset using PyTorch.

## Requirement
**Library** | **Version**
--- | ---
**Python** | **^2.7**
**PyTorch** | **^0.1.12** 
**Numpy** | **^1.12.0** 
**Pickle** |  *  

## Usage
### Download code:
```sh
git clone https://github.com/eugenelet/tensorflow-cifar-10-NiN

cd tensorflow-cifar-10-NiN
```

### Train cnn:
Batch size: 128

Prediction made on per epoch basis. 

161 epochs takes about 3h on GTX 1080.

```sh
python train.py
```

#### Make prediction:
```sh
python test.py
```

## Tensorboard
```sh
tensorboard --logdir=./tensorboard
```

## License
[Apache License 2.0](https://github.com/eugenelet/tensorflow-cifar-10-NiN/blob/master/LICENSE)

## Implementation Details
[My Blog](https://embedai.wordpress.com/2017/07/23/network-in-network-implementation-using-tensorflow/)
