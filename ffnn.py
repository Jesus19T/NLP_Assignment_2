import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser


unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        h1_Z = self.W1(input_vector)
        h1_Output = self.activation(h1_Z)

        # [to fill] obtain output layer representation
        h2_Z = self.W2(h1_Output)

        # [to fill] obtain probability dist.
        h2_Output = self.softmax(h2_Z)

        predicted_vector = h2_Output

        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        # print(elt["stars"]-1)
        # tra.append((elt["text"].split(),int(elt["stars"]-1)))
        tra.append((elt["text"].split(),int(elt["stars"])))

    for elt in validation:
        # val.append((elt["text"].split(),int(elt["stars"]-1)))
        val.append((elt["text"].split(),int(elt["stars"])))

    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    
    training_epoch_loss = []
    validation_epoch_loss = []

    training_epoch_accuracy = []
    validation_epoch_accuracy = []

    model = FFNN(input_dim = len(vocab), h = args.hidden_dim)
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
    print("========== Training for {} epochs ==========".format(args.epochs))
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16 
        # minibatch_size = 6
        # minibatch_size = 2

        training_minibatch_loss = []

        N = len(train_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                # print("input_vector")
                # print(input_vector)
                predicted_vector = model(input_vector)
                # predicted_vector = model.forward(input_vector)      #JESUS Added
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            training_minibatch_loss.append(loss.data)
            loss.backward()
            optimizer.step()
            
        training_epoch_loss.append(np.array(training_minibatch_loss).mean())
        training_epoch_accuracy.append(correct / total)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))


        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        minibatch_size = 16 
        # minibatch_size = 2

        validation_minibatch_loss = []

        N = len(valid_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                if gold_label == 3:
                    print("#############")
                    print(input_vector)
                    print("predicted_label", predicted_label)
                    if(predicted_label == gold_label):
                        print("Correct")
                    else:
                        print("Wrong")

                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            validation_minibatch_loss.append(loss.data)
        validation_epoch_loss.append(np.array(validation_minibatch_loss).mean())
        validation_epoch_accuracy.append(correct / total)
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))

    # write out to results/test.out

    #plot loss 
    from matplotlib import pyplot as plt
    plt.figure(1)
    plt.plot(training_epoch_loss, label='train_loss')
    plt.plot(validation_epoch_loss,label='val_loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.xticks(range(1,len(training_epoch_loss)))
    plt.xticks(np.arange(len(np.array(training_epoch_loss))), np.arange(1, len(np.array(training_epoch_loss))+1))
    plt.legend()

    plt.figure(2)
    plt.plot(training_epoch_accuracy, label='train_accuracy')
    plt.plot(validation_epoch_accuracy,label='val_accuracy')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    # plt.xticks(range(1,len(training_epoch_loss)))
    plt.xticks(np.arange(len(np.array(training_epoch_accuracy))), np.arange(1, len(np.array(training_epoch_accuracy))+1))
    plt.legend()

    plt.show()
    