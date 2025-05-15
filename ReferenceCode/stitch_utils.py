import torch
from torch import optim
from torch import nn
from torch.linalg import LinAlgError
from torch.utils import data
from collections import OrderedDict
import pandas as pd
import torchvision.transforms as transforms

import sys
import os

def train_model(model, train_loader, epochs, saveas, description, device, logtofile, 
                per_epoch_fn=None,                 
                milestones=None,
                lr=0.1):
    #print(f"{milestones=}")
    model.train()
    # define the loss function and the optimiser
    loss_function = nn.CrossEntropyLoss()
    # optimiser = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    optimiser=optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimiser, milestones=milestones)

    logtofile(f"Train Model on {description}")
    # the epoch loop: note that we're training the whole network
    for epoch in range(epochs):
        running_loss = 0.0
        for data in train_loader:
            # data is (images, labels) tuple
            # get the inputs and put them on the GPU
            inputs, labels = data                     

            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # zero the parameter gradients
            optimiser.zero_grad()
    
            # forward + loss + backward + optimise (update weights)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimiser.step()
            if milestones:
                scheduler.step()
    
            # keep track of the loss this epoch
            running_loss += loss.item()
        logtofile("Epoch %d, loss %4.2f" % (epoch, running_loss))
        per_epoch_fn() if per_epoch_fn else None
    logtofile('**** Finished Training ****')
    #save the trained model weights
    
    logtofile(f"{saveas=}")
    torch.save(model.state_dict(), saveas)


def get_layer_output_shape(model, layer_index, input_shape, device, type="ResNet18"):
    # Create a dummy input tensor with the given input size
    dummy_input = torch.rand(input_shape).to(device).unsqueeze(0)
    
    # This dictionary will store the shape of the output from the layer
    output_shape = {}

    # Define the hook function
    def hook(module, input, output):
        output_shape['shape'] = output.shape
        #print(output)
    print(f"get_layer_output_shape for {type=}")
    # Register the hook
    if type == "ResNet18" or type == "ResNet50":
        layer = list(model.children())[layer_index]
    elif type == "VGG19":
        layer = list(model.features.children())[layer_index]
    else:
        print("Unrecognised model type in get_layer_output_shape")
        assert False

    handle = layer.register_forward_hook(hook)
    model.eval()
    # Perform a forward pass with the dummy input
    with torch.no_grad():
        model(dummy_input)
    
    # Remove the hook
    handle.remove()
    
    return output_shape['shape']


def extract_children_from_index(model, layer_index, type="ResNet18"):
    if type == "ResNet18" or type == "ResNet50":       
        return nn.Sequential(*model.children())[layer_index:]
    elif type == "VGG19":        
        return nn.Sequential(*model.features.children())[layer_index:]
    else:
        print("Unrecognised model type in extract_children_from_index")
        assert False


def extract_children_to_index(model, layer_index, type="ResNet18"):
    # Add 1 to index so that the last layer indexed is included
    if type == "ResNet18" or type == "ResNet50":       
        return nn.Sequential(*model.children())[:layer_index+1]       
    elif type == "VGG19":        
        return nn.Sequential(*model.features.children())[:layer_index+1]
    else:
        print("Unrecognised model type in extract_children_from_index")
        assert False

def list_children(model):
    print(nn.Sequential(*model.children()))[:]


''' Only generate the activations once, so that the same values are used for train and test datasets.
Use torch.rand so that values are in the range [0,1] as this is more likely as a representation'''
def generate_activations(num_classes, representation_shape):
    activations = torch.empty((num_classes, ) + representation_shape)
    for c in range(num_classes):
        activations[c] = torch.rand(representation_shape)
    return activations


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, train, activations, noise=0.0, num_classes = 10):
        """
        Args:
            train (bool)        : for train=True set, return 60_000 samples, otherwise 10_000
            activations         : base set of activations: one per class. values in range [0,1]
            noise (float) = 0.0 : factor by which to scale randn values before adding to the base activations. 
            num_classes = 10    : standard for MNIST and CIFAR-10
        """
        self.train = train
        self.size = 60_000 if train else 10_000
        
        self.data = torch.empty((self.size,) + activations[0].shape)  # start off empty
        self.targets = torch.empty(self.size)  #, dtype=torch.int)  # Random targets
        
        for c in range(num_classes):
            #activations = torch.randn(self.shape)
            #print(activations.shape)
            # FIXME do I need to use clone()? This may not be producing the ball of noise I thought it was.
            self.data[c * self.size // num_classes : ((c+1) * self.size // num_classes)]    = activations[c].clone()
            self.targets[c * self.size // num_classes : ((c+1) * self.size // num_classes)] = c

        self.data += noise * torch.randn(self.data.shape)  # add a configurable amount of noise
        self.data = torch.clamp(self.data, 0, 1)           # and clamp to [0,1] range again so that it is batchnorm compatible
            
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = self.data[idx], int(self.targets[idx])
        return sample



class DynamicSyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, activations, noise=0.0, local_seed = 9):
        """
        Args:
            num_samples         : would use 60000/10000 for MNIST equivalent. May well require 1000 per class = 1 million for imagenet
            activations         : base set of activations: one per class. values in range [0,1]
            noise (float) = 0.0 : factor by which to scale randn values before adding to the base activations.             
            local_seed          : used to create random noise without upsetting the global generators
        """
        num_classes = len(activations)
        self.targets = torch.empty(num_samples)  #, dtype=torch.int)  # Random targets
        self.local_seed = local_seed
        self.noise = noise
        
        for c in range(num_classes):
            self.targets[c * num_samples // num_classes : ((c+1) * num_samples // num_classes)] = c

        self.activations = activations
        
        # Create a local random number generator
        self.local_rng = torch.Generator()
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        sample_class = int(self.targets[idx])
        # Seed the local generator according to this get request
        self.local_rng.manual_seed(self.local_seed + idx)

        # print(f"before adding noise: {sample_class=} {self.activations[sample_class][0]=}") 
        data = self.activations[sample_class].clone() 
        data += self.noise * torch.randn(data.shape, generator=self.local_rng)  # add a configurable amount of noise
        data = torch.clamp(data, 0, 1)                # and clamp to [0,1] range again so that it is batchnorm compatible 
        # print(f"after adding noise: {self.activations[sample_class][0]=}") 
             
        
        sample = data, sample_class
        return sample



# Produces a cut model, receiver only, with a stitch at the input
# after_layer_index = -1 used to duplicate model with matching naming
class RcvResNet18(nn.Module):
    def __init__(self, model, after_layer_index, input_image_shape, device):
        super(RcvResNet18, self).__init__()
        
        self.after_layer_index = after_layer_index
        if after_layer_index != -1:                
            self.output_shape = get_layer_output_shape(model, after_layer_index, input_image_shape, device)
            output_size = self.output_shape.numel()
            print(f"The shape of the output from layer {after_layer_index} is: {self.output_shape}, with {output_size} elements")
            self.stitch = nn.Sequential(OrderedDict([
              ('s_bn1',   nn.BatchNorm2d(self.output_shape[1])),
              ('s_conv1', nn.Conv2d(self.output_shape[1], self.output_shape[1], kernel_size=1, stride=1, padding=0)),
              ('s_bn2',   nn.BatchNorm2d(self.output_shape[1]))          
            ]))
        self.features = extract_children_from_index(model, after_layer_index + 1)
        self.features[-1] = nn.Identity()
        self.originalfc = model.fc
    
    def forward(self, x):
        if self.after_layer_index != -1:    
            z = self.stitch(x)
        else:
            z = x
        z = self.features(z)
        z = z.flatten(start_dim=1)
        out = self.originalfc(z)
        return out


# Produces a cut model, receiver only, with a stitch at the input
# after_layer_index = -1 used to duplicate model with matching naming
class RcvResNet50(nn.Module):
    def __init__(self, model, after_layer_index, input_image_shape, device):
        super(RcvResNet50, self).__init__()
        
        self.after_layer_index = after_layer_index
        if after_layer_index != -1:                
            self.output_shape = get_layer_output_shape(model, after_layer_index, input_image_shape, device, type="ResNet50")
            output_size = self.output_shape.numel()
            print(f"The shape of the output from layer {after_layer_index} is: {self.output_shape}, with {output_size} elements")
            self.stitch = nn.Sequential(OrderedDict([
              ('s_bn1',   nn.BatchNorm2d(self.output_shape[1])),
              ('s_conv1', nn.Conv2d(self.output_shape[1], self.output_shape[1], kernel_size=1, stride=1, padding=0)),
              ('s_bn2',   nn.BatchNorm2d(self.output_shape[1]))          
            ]))
        self.features = extract_children_from_index(model, after_layer_index + 1, type="ResNet50")
        self.features[-1] = nn.Identity()
        self.originalfc = model.fc
    
    def forward(self, x):
        if self.after_layer_index != -1:    
            z = self.stitch(x)
        else:
            z = x
        z = self.features(z)
        z = z.flatten(start_dim=1)
        out = self.originalfc(z)
        return out


# Produces a cut model, receiver only, with a stitch at the input
# -1 used to duplicate model with matching naming
class RcvVGG19(nn.Module):
    def __init__(self, model, after_layer_index, input_image_shape, device):
        super(RcvVGG19, self).__init__()
        
        self.after_layer_index = after_layer_index
        # For VGG19 the first layer of hierarchy is features, avgpool, classifier.
        # We will be slicing within features only
        if after_layer_index != -1:   
            self.output_shape = get_layer_output_shape(model, after_layer_index, input_image_shape, device, type="VGG19")
            output_size = self.output_shape.numel()
            print(f"The shape of the output from layer {after_layer_index} is: {self.output_shape}, with {output_size} elements")
                
            self.stitch = nn.Sequential(OrderedDict([
              ('s_bn1',   nn.BatchNorm2d(self.output_shape[1])),
              ('s_conv1', nn.Conv2d(self.output_shape[1], self.output_shape[1], kernel_size=1, stride=1, padding=0)),
              ('s_bn2',   nn.BatchNorm2d(self.output_shape[1]))          
            ]))
        
        self.features = extract_children_from_index(model, after_layer_index + 1, type="VGG19")
        self.avgpool = model.avgpool
        self.classifier = model.classifier
            
    def forward(self, x):
        if self.after_layer_index != -1:    
            z = self.stitch(x)
        else:
            z = x
        z = self.features(z)
        z = self.avgpool(z)
        f = torch.flatten(z, 1)
        out = self.classifier(f)
        return out
        

'''Takes first layers of send_model and stitches to last layers of rcv_model
Current implementation only stitches at same level of the networks, which must both be same architecture'''
class StitchedResNet18(nn.Module):
    def __init__(self, 
                 send_model, after_layer_index,
                 rcv_model,
                 input_image_shape, device):
        super(StitchedResNet18, self).__init__()
        
        self.output_shape = get_layer_output_shape(send_model, after_layer_index, input_image_shape, device)
        #output_size = self.output_shape.numel()
        print(f"The shape of the output from layer {after_layer_index} of send_model is: {self.output_shape}") #, with {output_size} elements")
                
        self.send_features = extract_children_to_index(send_model, after_layer_index)
        self.stitch = nn.Sequential(OrderedDict([
          ('s_bn1',   nn.BatchNorm2d(self.output_shape[1])),
          ('s_conv1', nn.Conv2d(self.output_shape[1], self.output_shape[1], kernel_size=1, stride=1, padding=0)),
          ('s_bn2',   nn.BatchNorm2d(self.output_shape[1]))          
        ]))
        self.rcv_features = extract_children_from_index(rcv_model, after_layer_index + 1)
        self.rcv_features[-1] = nn.Identity()
        self.original_rcv_fc = rcv_model.fc
    
    def forward(self, x):
        y = self.send_features(x)
        s = self.stitch(y)
        #z = z.reshape(z.shape[0], *self.output_shape[1:])
        z = self.rcv_features(s)
        z = z.flatten(start_dim=1)
        out = self.original_rcv_fc(z)
        return out


'''Takes first layers of send_model and stitches to last layers of rcv_model
Current implementation only stitches at same level of the networks, which must both be same architecture'''
class StitchedVGG19(nn.Module):
    def __init__(self, 
                 send_model, after_layer_index,
                 rcv_model,
                 input_image_shape, device):
        super(StitchedVGG19, self).__init__()
        
        self.output_shape = get_layer_output_shape(send_model, after_layer_index, input_image_shape, device, type="VGG19")
        #output_size = self.output_shape.numel()
        print(f"The shape of the output from layer {after_layer_index} of send_model is: {self.output_shape}") #, with {output_size} elements")
                
        self.send_features = extract_children_to_index(send_model, after_layer_index, type="VGG19")
        self.stitch = nn.Sequential(OrderedDict([
          ('s_bn1',   nn.BatchNorm2d(self.output_shape[1])),
          ('s_conv1', nn.Conv2d(self.output_shape[1], self.output_shape[1], kernel_size=1, stride=1, padding=0)),
          ('s_bn2',   nn.BatchNorm2d(self.output_shape[1]))          
        ]))

        self.rcv_features = extract_children_from_index(rcv_model, after_layer_index + 1, type="VGG19")
        self.avgpool = rcv_model.avgpool
        self.classifier = rcv_model.classifier
    
    def forward(self, x):
        y = self.send_features(x)
        s = self.stitch(y)
        #z = z.reshape(z.shape[0], *self.output_shape[1:])
        z = self.rcv_features(s)
        z = self.avgpool(z)
        z = z.flatten(start_dim=1)
        out = self.classifier(z)
        return out