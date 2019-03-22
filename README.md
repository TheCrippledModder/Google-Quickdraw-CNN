# Google-Quickdraw-CNN
Convolutional Neural Network that can classify over 50+ Million hand drawn images from the [Quickdraw](https://quickdraw.withgoogle.com/) dataset using [Tensorflow](https://www.tensorflow.org/)

## Requirements ##
[Quickdraw dataset](https://github.com/googlecreativelab/quickdraw-dataset) by Google
```bash
pip install tensorflow
```
## Model Summary ##

(The final two layers will change depending on the number of classes you choose to use)
```bash
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 98, 98, 64)        1792      
_________________________________________________________________
activation (Activation)      (None, 98, 98, 64)        0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 49, 49, 64)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 47, 47, 128)       73856     
_________________________________________________________________
activation_1 (Activation)    (None, 47, 47, 128)       0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 23, 23, 128)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 21, 21, 128)       147584    
_________________________________________________________________
activation_2 (Activation)    (None, 21, 21, 128)       0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 10, 10, 128)       0         
_________________________________________________________________
flatten (Flatten)            (None, 12800)             0         
_________________________________________________________________
dense (Dense)                (None, 5)                 64005     
_________________________________________________________________
activation_3 (Activation)    (None, 5)                 0         
=================================================================
Total params: 287,237
Trainable params: 287,237
Non-trainable params: 0
_________________________________________________________________
```
