# GNN_classification_noisy_label
# deep learning hackatlon 2024/2025

## Description of the solution

### Image teaser explaining the procedure
During the process was tested different models: gin, gin-virtual, gcn, gcn-virtual and transformer node. Also was utilized three tipes of Cross Entropy: the classic one, the Symmetric and the NoisyCrossEntropy.

### Overview of the method
In the hackaton we tested different hyperparameters found the best behavior with batch size: 32, 24 and learning rate between 0.0007 and 0.0005. During the exeperiments we observed a good performance with gin-virtual and model transformer, but the results was obtained with VGAE model. Another important part was the Cross Entropy used and the best behavior was got with NoisyCrossEntropy.

- Dataset A, C and D: NoisyCrossEntropy.
- Dataset C: SymmetriCrossEntropy

### main.py
The experiments was conducted on the kaggle notebook, simualting the args passing to the prompt through the kaggle interface.

### Train and test
To train the model the main needed the path of the train dataset, after the train compute an Ensemble using the best model of the four cycle. If the train path is not provided the code takes the best model of the dataset to do the Ensemble on this. The last was the prediction on the test dataset.

### Model used
To obtain the best performance was utilized the model.pth in model_for_train directory taken from the following github: https://github.com/cminuttim/Learning-with-Noisy-Graph-Labels-Competition-IJCNN_2025/tree/main, we employ on it a finetuning on each dataset using also an Ensemble, defined on the number of cycle decided, for the best model obtained. The best model was selected by the F1 score performance got during the train. 

### Information
Python version: 3.11.11

### Authors:
- Michael Corelli
- Kevin Giannandrea