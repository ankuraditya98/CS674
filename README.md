# CS674
CS674 Project: Speech-based 3D Face Animation

This project implements a simpler version of meshtalk for generating 3D animated face from speech. Existing approaches mainly focuses on upper face animation leading to it's limitation in their scalability. With the meshtalk we aim to not only animate the lips movement for a given speech but also focus on upper face part like eyebrows which are uncorrelated with the speech signal. The fundamental foundation of meshtalk methodology revolves around the utilization of a categorical latent space specifically tailored for facial animation purposes. This innovative latent space possesses the remarkable capability to disentangle information into two distinct categories: audio-correlated and audio-uncorrelated. The distinguishing factor lies in the novel cross-modality loss, enabling enhanced control over facial animation outcomes. For training, we use the mini data-set from an open source 'multiface' data-set containing data of two unique users/ mesh . 

# How to run the code

## Training
1. clone repository using `git clone`
2. cd into the project directory
3. download [data pickle files](https://drive.google.com/drive/folders/1N6P-v1yPJT0vDN5pO9qxFfK1nayBMG8B?usp=sharing) and save in project directory
4. run train.py

# External Libraries
* [multiface from facebook](https://github.com/facebookresearch/multiface)

# Member Contributions

#### Abdullah
* wrote encoders to extract facial features and fuse them with audio features - encoder.py (lines 94 - 147)
* data preprocessing - data_preprocessing.py
* aggregating multimodal dataset and ingesting it into the model - data_utils.py & pickling_face_templates.py
* installed multiface dataset
* partially worked on the loss function derivation


#### Ankur
* wrote audio encoder - encoder.py (lines 9-92)
* wrote decoder.py lines(5-60)
* Worked on Autoregressive Model - Contex_model.py

#### Alejandro
* wrote loss function definitions
* wrote training script
* worked on report and slides
* did the presentation
