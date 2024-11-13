### MirrorCSPNet
<img width="996" alt="Screenshot 2024-11-07 at 9 11 58 PM" src="https://github.com/user-attachments/assets/04af00af-ed66-47c9-aec1-34d2a9dc3db6">

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

Results

The attention-enhanced Faster R-CNN, combined with CSPDarkNet, FPN, and PAN, is expected to produce higher precision and recall compared to a standard Faster R-CNN model by focusing on the most relevant parts of the image during training and inference. Additionally, the integration of the Path Aggregation Network (PAN) helps to improve spatial information flow, further boosting model performance.

You can find sample output images and detailed training metrics in the Jupyter Notebook.

### Acknowledgements

This implementation builds upon the original Faster R-CNN architecture introduced by Shaoqing Ren, Kaiming He, Ross B. Girshick, and Jian Sun. 
Special thanks will be added once I have finalized this project.

## Contributing

Feel free to fork this repository, raise issues, or submit pull requests if you find areas for improvement. Contributions are welcome!


## License
This project is licensed under the MIT License. All ideas and concepts are free to use, provided that proper credit is given by referring to the original author. 

