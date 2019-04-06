# Denoising Autoencoder in Tensorflow 2.x

## Usage
Begin training the model with ```train.py```  
```
--learning_rate    n   (optional) Float: learning rate
--epochs           n   (optional) Integer: number of passes over the dataset
--batch_size       n   (optional) Integer: mini-batch size during training
--logdir          dir  (optional) String: log file directory
--keep_training        (optional) loads the most recently saved weights and continues training
--keep_best            (optional) save model only if it has the best loss so far
```
Sample the training model with ```sample.py```
```
--sample_size     n    (optional) Integer: number of samples to test
--model       filepath (required) String: path to a trained model (.h5 file)
--use_noise            (optional) adds noise to samples before feeding into the autoencoder
```

## References
Extracting and Composing Robust Features with Denoising Autoencoders (Vincent et al.) http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf 

