# Finger-Detection

### Project Structure

The software used for building the finger detection system is organized as follows:

- `data`
  - `Training-Dataset` : *Train Images-Masks and Augmented Images*
  - `Validation-Dataset` : *Valid Images-Masks and Augmented Images*
- `preprocessing`
  - `skin_color_histogram.ipynb` : *Color spaces analysis* 
  - `skin_color_characterization.ipynb` : *Skin detection* 
- `models`
  - `components.ipynb` : *Finger classification using morphological operators*
  - `data_augmentation.ipynb` : *Image augmentation*
  - `cnn.ipynb` : *Finger classification using Convolutional NN*
- `demo`
  - `finger_detection.py` : *Real Time System*



### Pipeline Exectuion

To allow third parties to reproduce the results, execute all the script in the following order:

1. `skin_color_histogram.ipynb`
2. `skin_color_characterization.ipynb`
3. `components.ipynb`



Alternatively, one can execute the `finger_detection.py` script and the detection & clasification system will pop-up in your laptop:

**pyhton3** finger_detection.py [CAMERA SOURCE]

The CAMERA SOURCE is 0 for laptop-cam or the corresponding port for a web-cam, default 1. The default configuration of the system perform at its best if hand are shown at a distance of $0.4-0.6$ meters. w.r.t the camera.



Notice that if `cnn.ipynb` is going to be executed, it is strongly recommended to make us of GPU's power as well as execute `data_augmentatio.ipynb` to generate more images.



### Requirements

All the work has been done using several libraries, one can install all of them executing:

**pip** install -r requirements.txt

