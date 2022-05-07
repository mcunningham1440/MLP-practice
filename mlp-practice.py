import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models, transforms
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif


class Network(nn.Module):
    """Defines the architecture for a multilayer perceptron.
    
    Inherits from the Module class in torch.nn.
    """
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(48, 24)
        self.fc2 = nn.Linear(24, 5)
        
        self.log_softmax = nn.LogSoftmax(dim=1)
            
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
        
        return x

    
class createDataset(Dataset):
    """Creates a Torch dataset.
    
    Inherits from the Dataset class in torch.utils.data.
    """
    
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        features = self.features[index]
        label = self.labels[index]
        return features, label

    
def trainModel(X_train, y_train, X_test, y_test, learn_rate, epochs):
    """Trains a multilayer perceptron on the data provided, plots the training curve, and returns its state dict.
    
    Args:
        X_train
            A tensor containing the training features
        y_train
            A tensor containing the training labels
        X_test
            A tensor containing the test features
        y_test
            A tensor containing the test labels
        learn_rate
            The learning rate for the model
        epochs
            The number of epochs to train the model for
    """
    
    
    # Creates a Torch dataset and dataloader from the training set
    
    train_dataset = createDataset(features=X_train, labels=y_train)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=515, shuffle=True)

    train_losses = []
    test_losses = []
    
    
    # Defines the model architecture, loss criterion, and optimizer
    
    model = Network()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    
    stop_epoch = False
    for e in range(epochs):
        # Trains the model
        
        running_loss = 0
        for features, labels in trainloader:
            optimizer.zero_grad()
            log_probs = model.forward(features)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() / len(trainloader)

        train_losses.append(running_loss)
        
        
        # Evaluates the model on the test set
        
        with torch.no_grad():
            model.eval()
            log_probs = model.forward(X_test)
            loss = criterion(log_probs, y_test)
            test_losses.append(loss.item())
        
        
        # Saves a copy of the model if the average test loss over the past 5 epochs is greater than the average 
        # test loss over the 5 epochs before. Empirically, this method seems to be good at saving the model near
        # the minimum of the test loss curve. Note that the model continues to train afterwards
        
        if not stop_epoch and len(test_losses) > 9:
            if np.mean(test_losses[len(test_losses)-5:]) > np.mean(test_losses[len(test_losses)-10:len(test_losses)-5]):
                stop_epoch = e
                saved_dict = model.state_dict()
    
    if not stop_epoch:
        saved_dict = model.state_dict()
    
    
    # Plots the model's performance throughout training on the training and test sets
    
    plt.plot(range(epochs), train_losses, label='Training loss')
    plt.plot(range(epochs), test_losses, label='Test loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    
    # If the model was saved because a minimum was detected, imports the saved model for evaluation and labels
    # the point at which the model was saved on the plot
    
    if stop_epoch:
        stop_label = "Model saved at epoch " + str(stop_epoch)
        plt.plot([stop_epoch, stop_epoch], [2, 0], color='gray', linestyle='-', linewidth=0.5)
        plt.text(stop_epoch, 2, stop_label, fontsize=8)
    else:
        stop_label = "Loss still falling, model saved at end of training"
        plt.text(0, 2, stop_label, fontsize=8)

    plt.legend()
    plt.show()
        
    return saved_dict


def classReport(saved_dict, X, y):
    """Evaluates a model's predictions on the given dataset and displays the multiclass classification report.
    
    Args:
        saved_dict
            The state dict for the model being evaluated
        X
            A tensor containing the features of the evaluation data
        y
            A tensor containing the labels of the evaluation data
    """
    
    model = Network()
    model.load_state_dict(saved_dict)
    
    with torch.no_grad():
        model.eval()
        log_probs = model.forward(X)

    print("Performance of model:")
    num_preds = np.array(torch.argmax(log_probs, dim=1))
    report = classification_report(y, num_preds, target_names=all_labels, zero_division=1)
    print(report)
    print("======================================================")

print("Loading labels...")


def pdToTorch(df):
    return torch.from_numpy(np.array(df, dtype=np.float32))


labels = np.genfromtxt('/Users/michaelcunningham/desktop/datasets/pm50.csv', delimiter='\t', dtype='str', usecols=1, skip_header=1)


# Creates an array of labels with the classes encoded as integers

all_labels = np.unique(labels)
label_indices = {}
for i in range(len(all_labels)):
    label_indices[all_labels[i]] = i
y = np.zeros(len(labels), dtype=int)
for i, value in enumerate(labels):
    y[i] = label_indices[value]

print("Labels loaded")
print()
      
print("Loading data. This may take a few moments...")

X = pd.read_csv("/Users/michaelcunningham/desktop/datasets/breast.csv", delimiter='\t')

print("Dataset loaded")
print()

X = X.transpose()
X.columns = X.iloc[0]
X = X.drop('Genes')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)


# Creates the features for the genes chosen by ANOVA and builds the model

print("Training ANOVA model...")
anova = SelectKBest(f_classif, k=48).fit(X_train, y_train)
X_train_anova = anova.transform(X_train)
X_test_anova = anova.transform(X_test)

X_train_anova, X_test_anova = pdToTorch(X_train_anova), pdToTorch(X_test_anova)

anova_dict = trainModel(X_train_anova, y_train, X_test_anova, y_test, 0.001, 2000)
classReport(anova_dict, X_test_anova, y_test)


# Creates the features from the pam50 genes (minus ORC6L and MAPT, which are not in the data) and builds the model

print("Training PAM50 model...")
pam50 = ["ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", "CDC20", "CDC6", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C", "KRT14", "KRT17", "KRT5", "MDM2", "MELK", "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "NDC80", "NUF2", "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS", "UBE2C", "UBE2T"]
X_train_pam50 = X_train[pam50]
X_test_pam50 = X_test[pam50]

X_train_pam50, X_test_pam50 = pdToTorch(X_train_pam50), pdToTorch(X_test_pam50)

pam50_dict = trainModel(X_train_pam50, y_train, X_test_pam50, y_test, 0.001, 2000)
classReport(pam50_dict, X_test_pam50, y_test)


# Creates the features from PCA and builds the model

print("Training PCA model...")
pca = PCA(n_components=48).fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

X_train_pca, X_test_pca = pdToTorch(X_train_pca), pdToTorch(X_test_pca)

pca_dict = trainModel(X_train_pca, y_train, X_test_pca, y_test, 0.001, 2000)
classReport(pca_dict, X_test_pca, y_test)


# Evaluates the performance of an ensemble of all 3 models

print("Performance of ensemble model:")

anova_model, pam50_model, pca_model = Network(), Network(), Network()
anova_model.load_state_dict(anova_dict)
pam50_model.load_state_dict(pam50_dict)
pca_model.load_state_dict(pca_dict)

with torch.no_grad():
    anova_log_probs = anova_model.forward(X_test_anova)
    pam50_log_probs = pam50_model.forward(X_test_pam50)
    pca_log_probs = pca_model.forward(X_test_pca)
    
anova_probs = torch.exp(anova_log_probs)
pam50_probs = torch.exp(pam50_log_probs)
pca_probs = torch.exp(pca_log_probs)


# Calcuates the ensemble probabilities of each class by averaging the probabilities from each model

ensemble_probs = (anova_probs + pam50_probs + pca_probs) / 3

num_preds = np.array(torch.argmax(ensemble_probs, dim=1))
report = classification_report(y_test, num_preds, target_names=all_labels, zero_division=1)
print(report)
