{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "#https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia\n",
    "#image preprocessing\n",
    "import cv2\n",
    "import numpy as np\n",
    "import os\n",
    "from PIL import Image\n",
    "from keras.preprocessing.image import load_img\n",
    "from keras.preprocessing.image import img_to_array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# window to viewport transformation\n",
    "def func(folder_path,folder_path1):\n",
    "    for img in os.listdir(folder_path):\n",
    "        copy=img\n",
    "        path1=(folder_path+'/'+img)\n",
    "        img=load_img(path=path1)\n",
    "        img=img_to_array(img)\n",
    "        first_dimension,second_dimension,third_dimension=img.shape\n",
    "        \n",
    "        xmin=0\n",
    "        ymin=0\n",
    "        xmax=first_dimension\n",
    "        ymax=second_dimension\n",
    "        \n",
    "        umin=0\n",
    "        vmin=0\n",
    "        umax=32\n",
    "        vmax=32\n",
    "        \n",
    "        sx=(umax-umin)/(xmax-xmin)\n",
    "        sy=(vmax-vmin)/(ymax-ymin)\n",
    "        \n",
    "        newimage=np.zeros(shape=(32,32,3),dtype=int)\n",
    "        \n",
    "        #for copying the image matrix in new image matrix\n",
    "        for i in range(0,32):\n",
    "            i1=((i-umin)/sx)+xmin\n",
    "            for j in range(0,32):\n",
    "                j1=((j-vmin)/sy)+ymin\n",
    "                i1=int(i1)\n",
    "                j1=int(j1)\n",
    "                newimage[i][j]=img[i1][j1]\n",
    "        path=(folder_path1+'/')\n",
    "        cv2.imwrite(os.path.join(path , copy), newimage)\n",
    "        cv2.waitKey(0)\n",
    "folder_path='chestxray//train//NORMAL'\n",
    "folder_path1='compressed32//train//NORMAL'\n",
    "func(folder_path,folder_path1)\n",
    "\n",
    "folder_path='chestxray//train//PNEUMONIA'\n",
    "folder_path1='compressed32//train//PNEUMONIA'\n",
    "func(folder_path,folder_path1)\n",
    "\n",
    "\n",
    "folder_path='chestxray//test//NORMAL'\n",
    "folder_path1='compressed32//test//NORMAL'\n",
    "func(folder_path,folder_path1)\n",
    "\n",
    "folder_path='chestxray//test//PNEUMONIA'\n",
    "folder_path1='compressed32//test//PNEUMONIA'\n",
    "func(folder_path,folder_path1)      "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#CNN architecture\n",
    "\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Convolution2D\n",
    "from keras.layers import MaxPooling2D\n",
    "from keras.layers import Flatten\n",
    "from keras.layers import Dense"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#initialize CNN\n",
    "classifier=Sequential()\n",
    "#feature detector or convolution layer\n",
    "classifier.add(Convolution2D(32,(3,3),input_shape=(32,32,3),activation='relu'))\n",
    "#maxpooling\n",
    "classifier.add(MaxPooling2D(pool_size=(2,2)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#feature detector and MAXpooling\n",
    "classifier.add(Convolution2D(32,(3,3),activation='relu'))\n",
    "classifier.add(MaxPooling2D(pool_size=(2,2)))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#full connection\n",
    "classifier.add(Flatten())\n",
    "classifier.add(Dense(output_dim=128,activation='relu'))\n",
    "classifier.add(Dense(output_dim=128,activation='relu'))\n",
    "classifier.add(Dense(output_dim=1,activation='sigmoid'))\n",
    "classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.preprocessing.image import ImageDataGenerator\n",
    "\n",
    "train_datagen = ImageDataGenerator(rescale=1./255,\n",
    "                                   shear_range=0.2,\n",
    "                                   zoom_range=0.2,\n",
    "                                   horizontal_flip=True)\n",
    "test_datagen = ImageDataGenerator(rescale=1./255)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "training_set = train_datagen.flow_from_directory('compressed32//train',\n",
    "                                                target_size=(32, 32),\n",
    "                                                batch_size=32,\n",
    "                                                class_mode='binary')\n",
    "                                                \n",
    "\n",
    "                                                "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "test_set = test_datagen.flow_from_directory('compressed32//test',\n",
    "                                             target_size=(32,32),\n",
    "                                             batch_size=32,\n",
    "                                             class_mode='binary')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "classifier.fit_generator(\n",
    "        training_set,\n",
    "        steps_per_epoch=5216,\n",
    "        epochs=5,\n",
    "        validation_data=test_set,\n",
    "        validation_steps=624)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "from keras.preprocessing.image import load_img\n",
    "falsepositive=0\n",
    "falsenegative=0\n",
    "truepositive=0\n",
    "truenegative=0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "folder_path='compressed32//test//NORMAL'\n",
    "for img in os.listdir(folder_path):\n",
    "        copy=img\n",
    "        path1=(folder_path+'/'+img)\n",
    "        img=load_img(path=path1)\n",
    "        img=img_to_array(img)\n",
    "        img=img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))\n",
    "        output=classifier.predict(img)\n",
    "        if(output[0][0]==1):\n",
    "            falsepositive=falsepositive+1\n",
    "        else:\n",
    "            truenegative=truenegative+1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "folder_path='compressed32//test//PNEUMONIA'\n",
    "for img in os.listdir(folder_path):\n",
    "        copy=img\n",
    "        path1=(folder_path+'/'+img)\n",
    "        img=load_img(path=path1)\n",
    "        img=img_to_array(img)\n",
    "        img=img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))\n",
    "        output=classifier.predict(img)\n",
    "        if(output[0][0]==1):\n",
    "            truepositive=truepositive+1\n",
    "        else:\n",
    "            falsenegative=falsenegative+1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "precision=truepositive/(truepositive+falsepositive)\n",
    "recall=truepositive/(truepositive+falsenegative)\n",
    "accuracy=(truepositive+truenegative)/(truepositive+truenegative+falsepositive+falsenegative)\n",
    "f1score=2*(precision*recall)/(precision+recall)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "classifier.save_weights('savedweights5.h5')\n",
    "classifier.load_weights('savedweights5.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
