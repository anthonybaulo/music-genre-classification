# Music Classification with a Convolutional Neural Network
This project explores the application of a CNN to audio using 2D Convolutions. This endeavor falls under the science of Music Information Retrieval (MIR), which has some well-known applications in Recommender Systems (Spotify) and Audio Identification (Shazam).<br>
[This video](https://youtu.be/lm5Pmkzw6Bs) gives a brief overview, similar to the README.

<a name="top"></a>
# Table of Contents
[Data](#data)<br>
[Technology](#tech)<br>
[Intro to CNNs](#CNNs)<br>
[Results](#results)<br>
[Conclusion](#conclusion)<br>
[Navigating the repo / Reproducing the results](#nav)<br>

---
---
<a name="data"></a>
## Data
The data comes from the [Free Music Archive](https://github.com/mdeff/fma) open-benchmark dataset.<br> 
I used the pre-defined "Small" subset, which offers 8000 30-second clips balanced over 8 root genres.

<a name="tech"></a>
## Technology
This project used Tensorflow2.0/Keras running on GPUs hosted on Amazon Web Services and employed the standard Python Data Science stack, with the inclusion of [Librosa](https://librosa.github.io/librosa/) for the audio conversion.
<p align="center">
  <img src="images/tech.png" width="60%" height="60%" />
</p>

<a name="CNNs"></a>
## Convolutional Neural Networks
CNNs are best known for their state of the art performance on image classification. To achieve this, they use a series of filters to scan the image for features, and at each layer of the network more complex features are found.

![https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)


The networks sees images as arrays of numbers, with each number representing a pixel value.
<p align="center">
  <img src="images/image_as_array.png" width="60%" height="60%" />
</p>

In order to use this network with audio, it must first be converted to a format similar to an image. The [melspectrogram](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0) offers such a format, where the numbers in the array represent decibel ratings at each timestep and frequency.

<p align="center">
  <img src="images/audio_as_array.png" width="70%"  />
</p>

<a name="results"></a>
## Results
### Rock vs. Hip-Hop
The first test was to see how the network distinguishes between Rock and Hip-Hop.

<p align="center">
  <img src="images/rock_v_hiphop.png"/>
</p>


Before training the model, the arrays were reduced to 2 principal components and plotted, showing that the genres cluster reasonably.

<p align="center">
  <img src="images/charts/PCA_rock_hiphop.png" width="40%"/>
</p>

After training on 800 examples of each genre, the model achieved 94% accuracy on a balanced test set of 200.

<p align="center">
  <img src="images/charts/model3_summary.png" height="80%" />
  <img src="images/charts/model3_cm.png" width="440px" />
</p>

### Rock vs. Hip-Hop vs. Instrumental
<p align="center">
  <img src="images/roc_v_hip_inst.png"/>
  When I introduced the more ambiguous genre of ‘Instrumental’ into the mix, there was more overlap in the plotting.
  <img src="images/charts/PCA_rock_hiphop_inst.png" width="40%" />
</p>

<p align="center">
After adding the new tracks to the network, accuracy dropped to 84% and struggled most with the instrumental genre.
  
  <img src="images/charts/model4_summary.png"/>
  <img src="images/charts/model4_cm.png" width="500px"/>
</p>

I listened to the misclassified clips to see what they sounded like. <br>
Since I can't embed the clips in the README, I'll just point out that the instrumental clips *do* resemble other genres. In one particular example where the model's prediction was 94% Hip-Hop, the "Instrumental" clip contained a sample of a human voice talking over a beat, which very much resembled Hip-Hop. 

These broad, subjective labels seem to be hard for the network to learn.

<a name="conclusion"></a>
## Conclusion
* High-level metadata can be extracted from an audio signal
* The CNN filters are able to learn the distinguishing features of broad genre classifications
* The network can only be as good as our subjective labeling system
* Looking into the misclassified examples can be very informative about your model and your data

## Next Steps
* Make scripts configurable
* Continue to add more genres, including lower-level sub-genres from the full dataset
* Replicate the architecture of a state of the art image classification model
* Compare to other networks, such as Conv1D to LSTM

---
---
<a name="nav"></a>
## Reproducing the results
*Specifically the three-genre model (Rock, Hip-Hip, Instrumental)

### Environment
1. Create conda environment from [linux_environment.yml](linux_environment.yml) or [mac_environment.yml](mac_environment.yml)

### Download and convert audio
2. cd into `src/` and run the following from inside the directory 
    1. [download_small.sh](src/download_small.sh)
    2. [convert.py](src/convert.py) [genres separated by space]
        1. `$ python convert.py Rock Hip-Hop Instrumental`

### Run model
3. From root directory, run [model4.py](model4.py)
    1. This will save the weights from the best epoch (monitoring val_loss), as well as plots of the training accuracy/loss and the confusion matrix into `models/`
