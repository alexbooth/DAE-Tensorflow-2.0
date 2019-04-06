# Denoising Autoencoder in Tensorflow 2.x

Demo of a DAE with eager execution in TF2 using the MNIST dataset. 

<p align="center"><img src="sample_images/z2-sampled.gif"></p>
The above GIF shows the latent space of a DAE trained on MNIST (using the model in this repo) that is sampled from the decoder at at epochs 0..60. The latent space above was bound to 2 dimensions with each in (-1, 1).

## Usage
Begin training the model with ```train.py```  
```
--learning_rate    n   (optional) Float: learning rate
--epochs           n   (optional) Integer: number of passes over the dataset
--batch_size       n   (optional) Integer: mini-batch size during training
--logdir          dir  (optional) String: log file directory
--keep_training        (optional) loads the most recently saved weights and continues training
--keep_best            (optional) save model only if it has the best loss so far
--help
```
Track training by starting Tensorboard and then navigate to ```localhost:6006``` in browser
```
tensorboard --logdir ./tmp/log/
```
Sample the training model with ```sample.py```  
Note: Do not run sample.py and train.py at the same time, tensorflow will crash.
```
--sample_size     n    (optional) Integer: number of samples to test
--model       filepath (required) String: path to a trained model (.h5 file)
--use_noise            (optional) adds noise to samples before feeding into the autoencoder
--help
```

## References
Extracting and Composing Robust Features with Denoising Autoencoders (Vincent et al.) http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf 

