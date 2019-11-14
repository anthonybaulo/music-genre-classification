# Music Classification with a Convolutional Neural Network
This project explores the application of a CNN to audio, using 2D Convolutions. This endeavor falls under the science of Music Information Retrieval (MIR), which has some well-known applications in Recommender Systems (Spotify) and Audio Identification (Shazam).

## Data
The data comes from the [Free Music Archive](https://github.com/mdeff/fma) open-benchmark dataset. I used the pre-defined "Small" subset, which offers 8000 30-second clips, balanced over 8 root genres.

## Convolutional Neural Networks
CNNs are best known for their state of the art performance on image classification. To achieve this, they use a series of filters to scan the image for features, and at each layer of the network, more complex features are found.

![https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)


The networks sees images as arrays of numbers, with each number representing a pixel value.
<p align="center">
  <img src="images/image_as_array.png" width="60%" height="60%" />
</p>

In order to use this network with audio, it must first be converted to a format similar to an image. The melspectrogram offers such a format, where the numbers in the array represent decibel ratings at each timestep and frequency.

<p align="center">
<img src="images/audio_as_array.png" width="70%"  />
</p>

## Rock vs. Hip-Hop
The first test will be to see how the network distinguishes between Rock and Hip-Hop.

<p align="center">
<img src="images/rock_v_hiphop.png"/>
</p>

Before training the model, the arrays can be reduced to 2 principal components and plotted, showing that the genres cluster.

<p align="center">
<img src="images/charts/PCA_rock_hiphop.png" width="40%"/>
</p>

After training on 800 examples of each genre, the model achieved 94% accuracy on a balanced test set of 200.

<p align="center">
<img src="images/charts/model3_summary.png" height="90%" />
<img src="images/charts/model3_cm.png" width="370px" />
</p>

## Rock vs. Hip-Hop vs. Instrumental
<p align="center">
  <img src="images/roc_v_hip_inst.png"/>
  <img src="images/charts/PCA_rock_hiphop_inst.png" width="40%" />
  <img src="images/charts/model4_summary.png"/>
  <img src="images/charts/model4_cm.png" width="500px"/>
</p>
