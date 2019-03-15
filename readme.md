# Pseudo Random Number Generation using Generative Adversarial Networks
This **exploratory** project seeks to assess the viability of generative 
adversarial networks (GANs) for the implementation of a
pseudo random number generator (CSPRNG). It draws inspiration from 2016 
work on bootstrapping encryption schemes using GANs: [Learning to Protect Communications
with Adversarial Neural Cryptography](https://arxiv.org/abs/1610.06918).

A short paper detailing the research, published at iWAISE 2018 (Dublin), is available on arXiv:
[Pseudo-Random Number Generation Using Generative Adversarial Networks](https://arxiv.org/abs/1810.00378)


## 1 - Setup and Executable
The easiest way to run the project is to pull the project's Docker image from
Docker Hub. This is because the project has software dependencies that are not
necessarily straightforward to setup, depending on the user's system. Furthermore,
the total size of the source code plus its dependencies far exceeds 50MB, so 
submitting everything via QMPlus is not possible. The pre-built Docker image is 
plug-and-play, and only requires the user install Docker. Check 
https://docs.docker.com/install/ for instructions on how to do this on your platform. 

Once Docker is correctly installed, execute the command

`docker run -i -t --rm marcellodebernardi/adversarial-csprng:main /bin/bash`

This will pull the Docker image from Docker Hub, start it in a new container, and
drop the user into an interactive shell inside the container. Refer to section 2
for what to do next. **It is strongly recommended to follow this approach.**

It is also possible to execute the system without Docker, but this requires manually
setting up the project dependencies. In order to do so, Python version `3.6` or 
higher is required. You can `pip install -r requirements.txt` to obtain the project's 
Python dependencies. Additional binaries (outside Python) on which the software depends
are listed in the Dockerfile. In general, the manual setup procedure is more or less
the same as that performed by Docker. 


## 2 - Training and Tweaking the Model
Once inside the running Docker container, run `python3 main.py` to train and 
evaluate the models. Parameters can be tweaked in the constants section at the 
top of the script, and some command-line options are supported. 

Available command line arguments:
```
-t              TEST MODE: runs model with reduced size and few iterations
-nodisc         SKIP DISCRIMINATIVE GAN: does not train discriminative GAN
-nopred         SKIP PREDICTIVE GAN: does not train predictive GAN
-long           LONG TRAINING: trains for 1,000,000 epochs
-short          SHORT TRAINING: trains for 10,000 epochs
-highlr         HIGH LEARNING RATE: increases the learning rate from default
-lowlr          LOW LEARNING RATE: decreases the learning rate from default
```

The system will not stop you from inputting contradictory cli arguments, such as
`python3 main.py -highlr -lowlr`. Common sense applies. In general,
**it is recommended to only train one of the models at once**, i.e. you should
add either the `-nopred` or `-nodisc` argument when executing the program. There
are currently some unresolved issues with the TensorFlow computational graph 
when training both models at the same time.


## 3 - Approaches to the Problem
The system trains and evaluates two separate models. These are two different
takes on the now popular *generative adversarial network* framework, though
with some major differences.

### 3.1 - Generative Adversarial Network with Discriminator
In this approach, a generator network produces output sequences that are served
as input to a discriminator, as per the usual GAN approach. The discriminator is
fed sequences from the generator as well as sequences obtained from a true randomness
source, and attempts to classify sequences as belonging to either category. The
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

The `components` package contains modules defining components of neural networks such 
as loss functions, activation functions, and other tensor operations. The `utils` package 
contains modules providing supporting functionality such as graph plotting and file output. 
The functionality provided by modules in this package is not crucial to the project 
at the conceptual level.


## 5 - Acknowledgements and License
This project is a final-year undergraduate dissertation carried out for the BSc Computer
Science at Queen Mary University of London. All rights to this work are held in accordance
to Queen Mary's policy on intellectual property of work carried out towards a degree.

Wherever this policy does not apply, or in any instance such that the policy does not make
a provision, the licence bundled with this repository shall apply.

The author would appreciate attribution for this work, if used anywhere.
