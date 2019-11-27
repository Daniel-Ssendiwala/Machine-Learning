#python2 compatibility
from __future__ import print_function
from __future__ import division

import numpy as np
import math
import skimage.transform
from keras.layers import Activation, Reshape, Dropout
from keras.models import Sequential
from keras.layers import AtrousConvolution2D, Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose
from keras.models import Sequential
from keras.utils import to_categorical
from keras.utils import plot_model
import keras.preprocessing.image
import keras.backend
import skimage
import colors
from PIL import Image
import urllib.request
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage.filters
from utils import *

# dataset: https://groups.csail.mit.edu/vision/datasets/ADE20K/browse.php/?dirname=training/
keras.backend.set_image_data_format('channels_last')

#weights #download from https://www.dropbox.com/s/9gnt784tu1m269n/dilatednet.npy?dl=0

def transform_digits_to_one_hot(digits):
    l = len(digits)
    nbclasses = 10
    new_y = np.zeros((l, nbclasses))
    for i, y_val in enumerate(digits):
        new_y[i][y_val] = 1
    
    return new_y
    
def convolution(image, kernel, stride):
    #p is always zero
    k = kernel.shape[0]
    h = image.shape[0]
    s = stride
    o = (h-k)//s +1
    out = np.zeros((o, o))
    
    # i,j = 0,0
    
    # while i +k <= h:
    #     while j+k <= h:
    #         out[i:i+k,j:j+k] = np.dot(kernel,image[i:i+k,j:j+k])
    #         j += s
    #     i += s
    #     j =0
    k2 = kernel[::-1,::-1]
    for x in range(o):
        for y in range(o):
            for kx in range(0, k):
                    for ky in range(0,k):
                        if x*s+kx >= 0 and y*s+ky >= 0 and y*s+ky < h and x*s+kx < h  :
                            # print(x,y,kx,ky ,x*s-kx,y*s-ky)
                            out[x][y] += k2[kx][ky] * image[x*s+kx][y*s+ky]
    # print(image)
    # print("kernel",kernel)
    # print("scipy ", scipy.signal.convolve2d(image, kernel, mode='valid', boundary='fill', fillvalue=0))
    # print("scipy",scipy.ndimage.filters.convolve(image, kernel, mode='constant', cval=0.0))
    # print("minout",out)
    scipy.ndimage.filters.convolve


    return out
    #return scipy.signal.convolve2d(in1, in2, mode='same', boundary='fill', fillvalue=0)

def max_pool(image,pool_size,stride):
    k =pool_size[0]
    p = math.ceil(float(image.shape[0])/k)
    out = np.zeros(( p, p))
    i,j = 0,0
    s = stride
    h = image.shape[0]
    while i //s < p:
        while j//s < p:
            out[i//s,j//s] = max(image[i:i+k,j:j+k].ravel())
            j += s
        i += s
        j =0
    return out

def sofmax(arr):
    arr -= np.max(arr)
    return np.exp(arr)/np.sum(arr)


def ReLU(x):
    max(0,x)


def img_to_array(width, height, url="http://web.mit.edu/graphicidentity/images/examples/tim-the-beaver-dont-1.png"):
    f = open('temp_im.jpg','wb')
    f.write(urllib.request.urlopen(url).read())
    f.close()
    im =  keras.preprocessing.image.load_img('temp_im.jpg', target_size=(width,height))
    im = im.crop((0,0,width,height))
    #im.show()
    plt.imshow(np.uint8(im))
    plt.show()   
    return np.array(im)
    
def label_image_to_one_hot(width, height,nbclasses, url="https://github.com/CSAILVision/sceneparsing/blob/master/sampleData/annotations/ADE_val_00000003.png?raw=true"):
    f = open('temp_label.jpg','wb')
    f.write(urllib.request.urlopen(url).read())
    f.close()
    im = keras.preprocessing.image.load_img('temp_label.jpg', target_size=(width,height), grayscale=True)
    print("label",np.uint8(im).shape)

    plt.imshow(np.float32(im),cmap='gray')
    plt.show()
    #im.show()
    im = np.array(im)
    return to_categorical(im, nbclasses)

def softmax_output_to_image(arr):
    w = int(np.sqrt(arr.shape[0]))
    img_arr = arr.argmax(1).reshape((w,w,1))
    
    im = keras.preprocessing.image.array_to_img(img_arr)
    #im.show()
    print(img_arr.shape)
    plt.imshow(np.float32(img_arr).reshape((w,w)), cmap='gray')
    plt.show()
    return img_arr

def get_model(input_width, input_height):
    model = Sequential()
    model.add(Convolution2D(64, (3, 3), activation='relu', name='conv1_1', input_shape=(input_width, input_height,3), padding='same'))
    model.add(Convolution2D(64, (3, 3), activation='relu', name='conv1_2', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(128, (3, 3), activation='relu', name='conv2_1', padding='same'))
    model.add(Convolution2D(128, (3, 3), activation='relu', name='conv2_2', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_1', padding='same'))
    model.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_2', padding='same'))
    model.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_3', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_1', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_2', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_3', padding='same'))

    # Compared to the original VGG16, we skip the next 2 MaxPool layers,
    # and go ahead with dilated convolutional layers instead

    model.add(Convolution2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_1', padding='same'))
    model.add(Convolution2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_2', padding='same'))
    model.add(Convolution2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_3', padding='same'))

    model.add(Convolution2D(4096, (7, 7), dilation_rate=(4, 4), activation='relu', name='fc6', padding='same'))
    # model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu', name='fc7',padding='same'))
    # model.add(Dropout(0.5))



    model.add(Convolution2D(151, (1, 1), activation='linear', name='fc-final151'))

    #model.add(Conv2DTranspose(1 , kernel_size=(16,16), strides=(8,8)  ,
    #                          activation='linear', padding='same',name="upsample",use_bias=False))
    
    _, curr_width, curr_height, curr_channels   = model.layers[-1].output_shape

    model.add(Reshape((curr_width * curr_height, curr_channels)))
    model.add(Activation('softmax'))

    return model

def load_weights(model):
    weights_data = np.load('./dilatednet.npy', encoding='latin1').item()
    for layer in model.layers:
        if layer.name in weights_data.keys():
            layer_weights = weights_data[layer.name]
            print(layer.name,layer_weights['weights'].shape)
            if 'biases' in layer_weights:
                layer.set_weights((layer_weights['weights'],
                                   layer_weights['biases']))
            else:
                print(layer.get_weights()[0].shape)
                s = layer_weights['weights'].shape
                layer.set_weights((layer_weights['weights'], ))
def train():
    # we will train on three images
    img1 = "https://github.com/CSAILVision/sceneparsing/blob/master/sampleData/images/ADE_val_00000001.jpg?raw=true"
    label1 = "https://github.com/CSAILVision/sceneparsing/blob/master/sampleData/annotations/ADE_val_00000001.png?raw=true"

    img2 = "https://github.com/CSAILVision/sceneparsing/blob/master/sampleData/images/ADE_val_00000002.jpg?raw=true"
    label2 = "https://github.com/CSAILVision/sceneparsing/blob/master/sampleData/annotations/ADE_val_00000002.png?raw=true"
    
    img3 = "https://github.com/CSAILVision/sceneparsing/blob/master/sampleData/images/ADE_val_00000003.jpg?raw=true"
    label3 = "https://github.com/CSAILVision/sceneparsing/blob/master/sampleData/annotations/ADE_val_00000003.png?raw=true"

    
    
    # lets 384 *384 size images
    width, height = 384,384
    model = get_model(width, height)
    print(model.summary())
    
    nbclasses = 151
    print("images")
    X_train = np.array( [img_to_array(width, height,img1), img_to_array(width, height,img2),
                         img_to_array(width, height,img3)] )
    print("labels")
    
    _,size,_   = model.layers[-1].output_shape
    print(size)
    width, height = (int(np.sqrt(size)),)*2
    
    Y_train = np.array( [label_image_to_one_hot(width, height,nbclasses, label1),
                         label_image_to_one_hot(width, height,nbclasses,label2), 
                         label_image_to_one_hot(width, height,nbclasses, label3)] )
    
    
    
    # set the optimizer and loss function. metrics is a list of metrics we would like to observe. 
    # They do not affect the training process but let us see how well we are doing 
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    # train the model
    # epochs is the number of times we would like to visit each image

    # load some pretrained weights
    load_weights(model)
    
    #model.fit(X_train, Y_train, 
    #      batch_size=3, epochs=1, verbose=1)
    

def predict():
    width, height = 384,384
    model = get_model(width, height)
    print(model.summary())
    
    # feel free to try other images
    #img1 = "https://github.com/CSAILVision/sceneparsing/blob/master/sampleData/images/ADE_val_00000003.jpg?raw=true"

    img1 = "http://groups.csail.mit.edu/vision/datasets/ADE20K//ADE20K_2016_07_26/images/training/a/art_school/ADE_train_00001703.jpg"

    img1 = "http://groups.csail.mit.edu/vision/datasets/ADE20K//ADE20K_2016_07_26/images/training/a/art_school/ADE_train_00001705.jpg"

    img1 = "http://groups.csail.mit.edu/vision/datasets/ADE20K//ADE20K_2016_07_26/images/validation/p/playground/ADE_val_00000707.jpg"

    #good image
    img1 =  "http://groups.csail.mit.edu/vision/datasets/ADE20K//ADE20K_2016_07_26/images/validation/p/podium/indoor/ADE_val_00001711.jpg"

    # img1 = "https://upload.wikimedia.org/wikipedia/en/2/24/Lenna.png"
    
    print("prediction input")
    #preprocess based on training data
    a = img_to_array(width, height,img1)
    mean = [109.5388, 118.6897, 124.6901];
    a = a[:,:,::-1] - mean
    
    showImg(a)
    X = np.array( [a])
    
    load_weights(model)
    Y = model.predict(X)
        
    width = int(np.sqrt(Y[0].shape[0]))

    print("prediction output grayscale")
    im = softmax_output_to_image(Y[0])

    #color and upsample the image
    print("prediction output color")
    ca = colorEncode(im.reshape((width,width,1)))
    ca = skimage.transform.rescale(ca, 8, mode='constant', cval=0)
    #im = keras.preprocessing.image.array_to_img(ca)
    #im.show()
    plt.imshow(ca)
    plt.show()
    return ca

def sobel(image):
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    gx = [None]*3
    gx[0] = convolution(image[:,:,0],kx,1)
    gx[1] = convolution(image[:,:,1],kx,1)
    gx[2] = convolution(image[:,:,2],kx,1)

    ky = np.array([[1,2,1],[0,0,0],[0,0,0]])
    gy = [None]*3
    gy[0] = convolution(image[:,:,0],ky,1)
    gy[1] = convolution(image[:,:,1],ky,1)
    gy[2] = convolution(image[:,:,2],ky,1)
    out = np.zeros((gx[0].shape[0],gx[0].shape[1],3))
    out[:,:,0] = gx[0] + gy[0]
    out[:,:,1] = gx[1] + gy[1]
    out[:,:,2] = gx[2] + gy[2]
    showImg(out)

def colorEncode(grayArr):
    print(grayArr.ravel())
    print(colors.colors[13])
    out = colors.colors[grayArr.ravel()].reshape(
        grayArr.shape[0:2] + (3,))
    print(out.shape)
    return out

if __name__ == '__main__':
    #train()
    predict()
    print(convolution(np.ones((8,8))*3, np.ones((2,2))*2, 1))
    im = np.int8(np.random.rand(8,8)*5)
    k = np.int8(np.random.rand(2,2)*2)
    assert np.array_equal(convolution(im,k,1),scipy.signal.convolve2d(im, k, mode='valid', boundary='fill', fillvalue=0))
    # assert np.array_equal(convolution(im,k,1), scipy.ndimage.filters.convolve(im, k, mode='constant', cval=0.0))
    #print(max_pool(np.ones((8,8)),(2,2), 1))
    img1 = "https://upload.wikimedia.org/wikipedia/en/2/24/Lenna.png"
    im = img_to_array(384,384,img1)
    sobel(im)

