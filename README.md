# Music Classification with a Convolutional Neural Network
Classifying audio clips by genre using a Convoultional Neural Network

## Intro
This project explores the application of a CNN to audio, using 2D Convolutions. This endeavor falls under the science of Music Information Retrieval (MIR), which has some well-known applications in Recommender Systems (Spotify) and Audio Identification (Shazam).

## Data
The data comes from the [Free Music Archive](https://github.com/mdeff/fma) open-benchmark dataset. I used the pre-defined "Small" subset, which offers 8000 30-second clips, balanced oer 8 root genres.

## Convolutional Neural Networks
![https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)
CNNs are best known for their state of the art performance on image classification. To achieve this, they use a series of filters to scan the image for features, and at each layer of the network, more complex features are found. <br>
The networks sees images as arrays of numbers, with each number representing a pixel value.

<img src="images/image_as_array.png" width="50%" height="50%" />

<img src="images/charts/spec_rock.png" width="50%" height="50%" />
<img src="images/charts/spec_hiphop.png" width="50%" height="50%" />
<img src="images/charts/PCA_rock_hiphop.png" width="50%" height="50%" />
<img src="images/charts/model3_summary.png" width="80%" height="80%" />
<img src="images/charts/model3_cm.png" width="40%" height="40%" />
