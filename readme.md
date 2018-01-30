# Cryptographically Strong Pseudo Random Number Generation using Generative Adversarial Networks
This **exploratory** project seeks to assess the viability of generative 
adversarial networks (GANs) for the implementation of a cryptographically 
secure pseudo random number generator (CSPRNG). It draws inspiration from recent 
work on bootstrapping encryption schemes using GANs: [Learning to Protect Communications
with Adversarial Neural Cryptography](https://arxiv.org/abs/1610.06918).



## 1 - Setup
The system targets `Python 3.6.2`, with module import requirements outlined
in `requirements.txt`. More in-depth dependency management will be added in
the future. A special dependency is `GraphViz`, which requires externally
installing its binaries. These can be found on the `GraphViz` website.


## 2 - Running and Tweaking the Model
Run `main.py` to train and evaluate the models. Parameters can be tweaked
in the constants section at the top of the script.


## 3 - Approaches to the Problem
The system trains and evaluates three separate models. Two are variation of the
*generative adversarial network* framework, and two are a single neural network.
The approaches, briefly outlined, are:

### 3.1 - Generative Adversarial Network
In this approach, a generator network produces output sequences that are served
as input to a discriminator, as per the usual GAN approach. The discriminator is
fed sequences from the generator as well as sequences obtained from a true randomness
source, and attempts to classify sequences as belonging in either category. The
generator attempts to maximize the error of the discriminator, and as thus should
learn the distribution of "true" randomness source used.

### 3.2 - Generative Adversarial Network with Predictor
A generator network produces output sequences that are served as input to a predictor,
with exception of the last value in the sequence. The predictor attempts to output
this value, and its loss function rewards it the closer to the value it gets. The
generator attempts to maximize the error of the predictor. The notion of unpredictability
equating with statistical randomness comes from the universality of the next bit
test, which states that a binary sequence passing the next bit test passes all
polynomial-time statistical tests. It stands to reason that the principle can be applied
to sequences of real numbers.


## 4 - Project Structure
The high-level functionality is in `main.py`. 

The `models` package contains modules defining components of neural networks such 
as loss functions, activation functions, and . The `utils` package contains 
modules either providing supporting functionality such as graph plotting or convenience 
functions, or abstracting the details of defining and compiling the Keras models. In the 
latter case, the functionality provided by this package is not crucial to the project 
at the conceptual level, and serves only to provide further abstraction over the Keras API.


## 5 - To-dos
3. Evaluation using statistical tests
4. Evaluation using NIST
7. Saving and restoring models
