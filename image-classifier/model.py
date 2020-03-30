from torchvision import models
from torch import nn, optim
import numpy as np
from PIL import Image
from collections import OrderedDict
import torch

class Classifier():

    def __init__(self):
        self.device = None
        self.model = None
    
    def prepareModel(self, device, hidden_units, lr, architecture = 'vgg16'):
        #self.model = models[architecture](pretrained=True)
        self.model = getattr(models, architecture)(pretrained=True)
        self.device = device
        for param in self.model.parameters():
            param.requires_grad = False
            
        number_inputs = self.model.classifier[0].in_features
        print("number_inputs ", number_inputs,hidden_units)
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(number_inputs, hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.4)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        self.model.classifier = classifier
        self.model = self.model.to(self.device)
        
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=lr)
        print("CLASSIFIER MODEL :", self.model )
        
    def train(self, n_epochs, loaders, use_cuda, save_path):
        """returns trained model"""
        #print( "use_cuda" , use_cuda)
        valid_loss_min = np.Inf 
        
        for epoch in range(1, n_epochs+1):
            # initialize variables to monitor training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
            
            ###################
            # train the model #
            ###################
            self.model.train()
            for batch_idx, (data, target) in enumerate(loaders['train']):
                # move to GPU
                
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                ## find the loss and update the model parameters accordingly
                ## record the average training loss, using something like
                ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
                
                # reset grads to zero
                self.optimizer.zero_grad()
                # model output
                output = self.model(data)
                # loss
                loss = self.criterion(output, target)
                # back propagation step
                loss.backward()
                # update the weights
                self.optimizer.step()
                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
                #pretty print status
                if batch_idx % 50 == 0:
                    print('Epoch {}, Batch {} training_loss: {:.6f}'.format(epoch, batch_idx+1, train_loss))
                
            ######################    
            # validate the model #
            ######################
            self.model.eval() # important 
            for batch_idx, (data, target) in enumerate(loaders['valid']):
                # move to GPU
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                ## update the average validation loss
                output = self.model(data)
                loss = self.criterion(output, target)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
                if batch_idx % 50 == 0:
                    print('Epoch {}, Batch {} validation_loss: {:.6f}'.format(epoch, batch_idx+1, train_loss))
            ## TODO: save the model if validation loss has decreased
            if valid_loss < valid_loss_min:
                torch.save(self.model.state_dict(), save_path)
                print('Validation loss decreased from ({:.6f} --> {:.6f}). Hence, saving model ...'.format(
                valid_loss_min,
                valid_loss))
                valid_loss_min = valid_loss
        # return trained model
        return self.model


    # TODO: Do validation on the test set
    def test(self, loaders, use_cuda):

        # monitor test loss and accuracy
        test_loss = 0.
        correct = 0.
        total = 0.

        self.model.eval()
        for batch_idx, (data, target) in enumerate(loaders['test']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = self.model(data)
            # calculate the loss
            loss = self.criterion(output, target)
            # update average test loss 
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
            # convert output probabilities to predicted class
            pred = output.data.max(1, keepdim=True)[1]
            # compare predictions to true label
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)
                
        print('Test Loss: {:.6f}\n'.format(test_loss))

        print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
            100. * correct / total, correct, total))
        
    def predict(self, image_path, cat_to_name, topk=1):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''

        # TODO: Implement the code to predict the class from an image file
        image = self.process_image(image_path)
        image = image.unsqueeze_(0)
        image = image.cuda().float()

        self.model.eval()

        with torch.no_grad():
            output = self.model(image)
            prob, ids = torch.topk(output, topk)
            ids = np.array(ids)            
            idx_to_class = {val:key for key, val in self.model.class_to_idx.items()}
            classes = [idx_to_class[id] for id in ids[0]]


            classnmaes = []
            for cls in classes:
                classnmaes.append(cat_to_name[str(cls)])

            return prob, classnmaes 
        
    def saveModel(self, filename, image_datasets):
        # TODO: Save the checkpoint 
        self.model.class_to_idx = image_datasets['train'].class_to_idx
        checkpoint = {'class_to_idx': self.model.class_to_idx,
                    'state_dict': self.model.state_dict(),
                    'classifier': self.model.classifier,
                    'model': self.model}


        torch.save(checkpoint, 'checkpoint.pt')
        
        
    # TODO: Write a function that loads a checkpoint and rebuilds the model
    def loadCheckpoint(self, checkpoint = "checkpoint.pt"):
        checkpoint = torch.load(checkpoint)
        self.model = checkpoint['model']
        for param in self.model.parameters(): 
            param.requires_grad = False

        self.model.class_to_idx = checkpoint['class_to_idx']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.classifier = checkpoint['classifier']
     
    
    def process_image(self, image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''

        # TODO: Process a PIL image for use in a PyTorch model
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image.thumbnail((256,256), Image.LANCZOS)

        width = image.size[0]   
        height = image.size[1]

        left = (width - 224)/2
        top = (height - 224)/2
        right = (width + 224)/2
        bottom = (height + 224)/2

        image = image.crop((left, top, right, bottom))

        image_array = np.array(image) / 255

        image = (image_array - mean) / std
        #image = image.transpose((2,1,1))
        image = image.transpose((2,0,1))
        #image = image.transpose((1, 2, 0))

        return torch.from_numpy(image)