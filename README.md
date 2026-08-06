# Crop-Disease-Classification-using-Transfer-Learning
Multi-class plant disease classification using EfficientNetB4 transfer learning and TensorFlow/Keras


Used pre-trained model: EfficientNetB4
Input shape:160 x 160 x 3 for RGB images
batch size: 32

stage 1 :Pre trained model for feature extraction-Base model was initially frozen to train only the new classifier head
stage 2: Fine Tuning on Pre trained model-First 100 layers were frozen and rest of the layers after were allowed to learn and adjust their parameters to adapt high level features

Dataset Description :Plant Leaf disease dataset with augmentation
		     39 different classes
		     61,486 images 
		     six different augmentation methods [image flipping, Gamma correction, noise injection, PCA color augmentation, rotation, and Scaling]

Reference Tutorial: https://www.tensorflow.org/tutorials/images/transfer_learning
