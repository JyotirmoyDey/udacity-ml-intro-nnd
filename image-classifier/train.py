import torch
from argparse import ArgumentParser
from model import Classifier
from torchvision import transforms, datasets
import os

def main():
    
    parser = ArgumentParser()
    parser.add_argument("--d_dir", type=str, help="directory to load the dataset from")
    parser.add_argument("--save_dir", type=str, default='checkpoint.pt', help="directory to save the model")
    parser.add_argument("--arch", type=str, default='vgg16', help="model architecture")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--hidden_units", default=128, type=int, help="hidden units to teh network")
    parser.add_argument("--gpu", type=bool, default=True, help="gpu required")
    parser.add_argument("--epochs", type=int, default=1, help="epochs")

    args = vars(parser.parse_args())

    train_dir = args["d_dir"] + '/train'
    valid_dir = args["d_dir"] + '/valid'
    test_dir = args["d_dir"] + '/test'

    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5], 
                                                                [0.5, 0.5, 0.5])]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(args["d_dir"], x), data_transforms[x]) 
                      for x in ['train', 'valid', 'test']}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
                                                                 for x in ['train', 'valid', 'test']}
    if torch.cuda.is_available() and args["gpu"] is True:
        device = torch.device("cuda:0")
        use_cuda = True
    else:
        device = torch.device("cpu")
        use_cuda = False
    print("DEVICE: ", device)

    classifier = Classifier()
    classifier.prepareModel(device, args["hidden_units"], args["lr"], args["arch"])

    classifier.train(args["epochs"], dataloaders, use_cuda, args["save_dir"])
    classifier.test(dataloaders, use_cuda)

    classifier.saveModel('checkpoint.pt', image_datasets)
    saved_model = classifier.loadCheckpoint()
    print("SAVED MODEL :", saved_model)

if __name__ == '__main__':
    main()
