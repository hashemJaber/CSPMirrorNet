### CSPMirrorNet
<p width="20%" align="center">
<img width="996" alt="Screenshot 2024-11-07 at 9 11 58 PM" src="https://github.com/user-attachments/assets/04af00af-ed66-47c9-aec1-34d2a9dc3db6" width="20%" align="center">
</p>
This repository aims to introduce a new proposed backbone to which we call CSPMirrorNet, CSPMirrorNet aims to add a Horizontal expansion of feature map dimensions to increase gradient representation, employing a siamese network-like structure, and implementing a concept of early feature map cross sectionality, all the while paying homage to CSPNet.

### Overview

CSPDenseNet is a widely-used deep learning backbone architecture for object detection tasks.  By adding advances such as a Horizontal expansion of feature map dimensions to increase gradient representation, employing a siamese network-like structure, and implementing a concept of early feature map cross sectionality backbone architectures to this model, we aim to increase the feature map representation while still attempting to limit the propogation time and memory consumption, but with a bigger aim of increasing performance.

The repository contains two main components:

Faster_RCNN_Attention.ipynb: A Jupyter Notebook that walks through the development, training, and evaluation of the model. It provides visualizations and explanations that allow for an interactive exploration of the Faster R-CNN with attention.

Faster_RCNN_Attention.py: The Python script containing the core implementation of the Faster R-CNN model with attention mechanisms, including the CSPDarkNet backbone, Feature Pyramid Networks (FPN), Path Aggregation Network (PAN), and Region Proposal Network (RPN).

### Requirements
To run the code in this repository, you will need the following:

- Python 3.7 or higher
- PyTorch
- OpenCV (for image processing)
- NumPy
- Matplotlib (for visualizations)
- Jupyter Notebook (optional, for the interactive `.ipynb` file)otebook (optional, for the interactive .ipynb file)
'''

You can install the necessary dependencies using the following command:
```bash
pip install -r requirements.txt 
```

> [!NOTE]  
> This is still a in-progress project so the requirments.txt will be made available once this project is done.


### Features

Attention Mechanism: Integrates attention layers into the Faster R-CNN backbone to enhance feature representation.

CSPDarkNet Backbone: Uses a CSPDarkNet-inspired backbone to improve gradient flow and reduce computational complexity while maintaining high accuracy.

Feature Pyramid Network (FPN) and Path Aggregation Network (PAN): Enhances the feature extraction process, allowing the model to effectively handle multi-scale objects.

Interactive Notebook: Provides step-by-step guidance through the training process, including visual outputs to illustrate the attention effects on feature maps.

Customizable Backbone: Allows for customization of the underlying convolutional neural network.

### Proposals and Innovations

## Proposal 1
- **Hpothesis**: Using average pooling, permutes, and channel manipulation followed by summation/concatenation to increase feature size representations without adding significant computational expense.

## Proposal 2
- **Hpothesis**: Horizontal expansion of feature map dimensions to increase gradient representation, employing a Siamese network-like structure for CSPNet to enhance feature richness while reducing computation.

## Proposal 3
- **Hpothesis**: Cross-section feature sharing within CSPNet to maintain edge information, enhancing the overall feature representation.
  
## Proposal 4
- **Hpothesis**: Attention, not there yet.

## Usage

### Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/hashemJaber/Faster_RCNN_Attention.git
cd Faster_RCNN_Attention
```

Run the Jupyter Notebook

Launch Jupyter Notebook to interactively explore the Faster R-CNN model with attention mechanisms:

jupyter notebook Faster_RCNN_Attention.ipynb

```bash
jupyter notebook Faster_RCNN_Attention.ipynb
```
Run the Python Script



You can also directly run the Python script for training or inference:
> [!NOTE]  
> This is still a in-progress project so the correct code will be made available once this project is done.


```bash
python Faster_RCNN_Attention.py
```

### Project Structure
Faster_RCNN_Attention.ipynb: A Jupyter Notebook providing a tutorial-style implementation of the model.

Faster_RCNN_Attention.py: The main script containing the model definition, including CSPDarkNet, FPN, PAN, SPP, RPN, and detection heads.

requirements.txt: Lists the dependencies required to run the project.

> [!CAUTION]
>  Running/Training the model on MS-COCO 2017 via GPU will cause errors hence for now RUN ON CPU.

## Results
# MS-COCO 2017

| Metric                      | IoU       | Area    | Max Dets | Value  |
|-----------------------------|-----------|---------|----------|--------|
| Average Precision (AP)      | 0.50:0.95 | all     | 100      | 0.000  |
| Average Precision (AP)      | 0.50      | all     | 100      | 0.000  |
| Average Precision (AP)      | 0.75      | all     | 100      | 0.000  |
| Average Precision (AP)      | 0.50:0.95 | small   | 100      | 0.000  |
| Average Precision (AP)      | 0.50:0.95 | medium  | 100      | 0.000  |
| Average Precision (AP)      | 0.50:0.95 | large   | 100      | 0.000  |
| Average Recall (AR)         | 0.50:0.95 | all     | 1        | 0.000  |
| Average Recall (AR)         | 0.50:0.95 | all     | 10       | 0.000  |
| Average Recall (AR)         | 0.50:0.95 | all     | 100      | 0.001  |
| Average Recall (AR)         | 0.50:0.95 | small   | 100      | 0.000  |
| Average Recall (AR)         | 0.50:0.95 | medium  | 100      | 0.001  |
| Average Recall (AR)         | 0.50:0.95 | large   | 100      | 0.002  |
### Detection Results

| Class              | Images | Instances | Box(P)  | R       | mAP50  | mAP50-95 |
|--------------------|--------|-----------|---------|---------|--------|----------|
| all                | 5000   | 36335     | 0.000117 | 0.00484 | 9.32e-05 | 2.29e-05 |
| person             | 2693   | 10777     | 0.0016   | 0.215   | 0.0023   | 0.000565 |
| bicycle            | 149    | 314       | 0        | 0       | 0        | 0        |
| car                | 535    | 1918      | 0        | 0       | 0        | 0        |
| motorcycle         | 159    | 367       | 0        | 0       | 0        | 0        |
| airplane           | 97     | 143       | 0        | 0       | 0        | 0        |
| bus                | 189    | 283       | 0        | 0       | 0        | 0        |
| train              | 157    | 190       | 0        | 0       | 0        | 0        |
| truck              | 250    | 414       | 0        | 0       | 0        | 0        |
| boat               | 121    | 424       | 0        | 0       | 0        | 0        |
| traffic light      | 191    | 634       | 0        | 0       | 0        | 0        |
| fire hydrant       | 86     | 101       | 0        | 0       | 0        | 0        |
| stop sign          | 69     | 75        | 0        | 0       | 0        | 0        |
| parking meter      | 37     | 60        | 0        | 0       | 0        | 0        |
| bench              | 235    | 411       | 0        | 0       | 0        | 0        |
| bird               | 125    | 427       | 0        | 0       | 0        | 0        |
| cat                | 184    | 202       | 0        | 0       | 0        | 0        |
| dog                | 177    | 218       | 0        | 0       | 0        | 0        |
| horse              | 128    | 272       | 0        | 0       | 0        | 0        |
| sheep              | 65     | 354       | 0        | 0       | 0        | 0        |
| cow                | 87     | 372       | 0        | 0       | 0        | 0        |
| elephant           | 89     | 252       | 0        | 0       | 0        | 0        |
| bear               | 49     | 71        | 0        | 0       | 0        | 0        |
| zebra              | 85     | 266       | 0        | 0       | 0        | 0        |
| giraffe            | 101    | 232       | 0        | 0       | 0        | 0        |
| backpack           | 228    | 371       | 0        | 0       | 0        | 0        |
| umbrella           | 174    | 407       | 0        | 0       | 0        | 0        |
| handbag            | 292    | 540       | 0        | 0       | 0        | 0        |
| tie                | 145    | 252       | 0        | 0       | 0        | 0        |
| suitcase           | 105    | 299       | 0        | 0       | 0        | 0        |
| frisbee            | 84     | 115       | 0        | 0       | 0        | 0        |
| skis               | 120    | 241       | 0        | 0       | 0        | 0        |
| snowboard          | 49     | 69        | 0        | 0       | 0        | 0        |
| sports ball        | 169    | 260       | 0        | 0       | 0        | 0        |
| kite               | 91     | 327       | 0        | 0       | 0        | 0        |
| baseball bat       | 97     | 145       | 0        | 0       | 0        | 0        |
| baseball glove     | 100    | 148       | 0        | 0       | 0        | 0        |
| skateboard         | 127    | 179       | 0        | 0       | 0        | 0        |
| surfboard          | 149    | 267       | 0        | 0       | 0        | 0        |
| tennis racket      | 167    | 225       | 0        | 0       | 0        | 0        |
| bottle             | 379    | 1013      | 0        | 0       | 0        | 0        |
| wine glass         | 110    | 341       | 0        | 0       | 0        | 0        |
| cup                | 390    | 895       | 0        | 0       | 0        | 0        |
| fork               | 155    | 215       | 0        | 0       | 0        | 0        |
| knife              | 181    | 325       | 0        | 0       | 0        | 0        |
| spoon              | 153    | 253       | 0        | 0       | 0        | 0        |
| bowl               | 314    | 623       | 0.00268  | 0.00803 | 0.0014   | 0.000339 |
| banana             | 103    | 370       | 0        | 0       | 0        | 0        |
| apple              | 76     | 236       | 0        | 0       | 0        | 0        |
| ...                | ...    | ...       | ...      | ...     | ...      | ...      |

### Model and Configuration Details
* Model: simplified Yolov8n with C2f replaced with CSPMirrorNet
* Training Device: CPU 
* Dataset: MS-COCO (No Data Augmentation or dropout)
* Training Configuration:
    * Learning Rate: Standard
    * Batch Size: 16
    * Optimizer: ADAM
    * Number of Epochs: 0, took 15 hours of CPU time to complete 30% of an epoch

# CIFAR-10
<p align="center">
  <img src="https://github.com/user-attachments/assets/2db8308d-9735-431b-b3ad-057027e49fa2" alt="Training and Validation Accuracy" width="45%">
  <img src="https://github.com/user-attachments/assets/94d17826-0a82-4562-8aec-fefe2e61111f" alt="Training and Validation Loss" width="45%">
</p>


### Model and Configuration Details
- **Model**: CSPMirrorNet53
- **Training Device**: NVIDIA A100 GPU
- **Dataset**: CIFAR-10 (No Data Augmentation or dropout)
- **Training Configuration**:
  - **Learning Rate**: Standard
  - **Batch Size**: 128
  - **Optimizer**: SGD @lr=0.1, @momentum=0.9, @weight_decay=5e-4
  - **scheduler**: @step_size=25, @gamma=0.5
  - **Number of Epochs**: 100
  - **Gamma**: Set to 0.8
 

You can find sample output images and detailed training metrics in the Jupyter Notebook.

### Acknowledgements

This implementation builds upon the original Faster R-CNN architecture introduced by Shaoqing Ren, Kaiming He, Ross B. Girshick, and Jian Sun. 
Special thanks will be added once I have finalized this project/research.

## Contributing

Feel free to fork this repository, raise issues, or submit pull requests if you find areas for improvement. Contributions are welcome!


## License
This project is licensed under the MIT License. All ideas and concepts are free to use, provided that proper credit is given by referring to the original author. 

