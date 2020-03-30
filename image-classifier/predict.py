from model import Classifier
from argparse import ArgumentParser
import json
import torch
from PIL import Image


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default='checkpoint.pt',  help="checkpoint file")
    parser.add_argument("--path_image", type=str, help="image path file")
    parser.add_argument("--top_k", type=int, default=1, help="top k probablities")
    parser.add_argument("--category_names", type=str, default='cat_to_name.json', help="category mapping to real names")
    parser.add_argument("--gpu", type=bool, default=True, help="gpu required")

    args = vars(parser.parse_args())
    
    with open(args["category_names"], 'r') as f:
        cat_to_name = json.load(f)
    
    if torch.cuda.is_available() and args["gpu"] is True:
        device = torch.device("cuda:0")
        use_cuda = True
    else:
        device = torch.device("cpu")
        use_cuda = False
    print("DEVICE: ", device)
    
    model = Classifier()
    model.loadCheckpoint(checkpoint=args["checkpoint"])
    
    image_pil = Image.open(args["path_image"])
    
    print(model.predict(image_pil, cat_to_name, topk=5))
                   
if __name__ == '__main__':
    main()                