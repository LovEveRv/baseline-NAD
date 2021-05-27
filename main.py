import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import classification_report
from data_loader import CatDogLoader, WasteLoader, GTSRBLoader
from models import VGG, VGG_bn, ViT
from tqdm import tqdm


def get_model(args, num_classes, ckpt):
    # if args.model == 'VGG':
    #     model = VGG(num_classes)
    # elif args.model == 'VGG_bn':
    #     model = VGG_bn(num_classes)
    # elif args.model == 'vit':
    #     model = ViT(num_classes)
    # else:
    #     raise NotImplementedError()
    
    model = VGG(num_classes)

    dct = torch.load(ckpt)
    model.load_state_dict(dct['model'])
    return model


def run_train(args, s_model, t_model, loader):
    optimizer = optim.Adam(s_model.parameters(), args.lr)

    s_model.train()
    t_model.eval()

    criterion_cls = nn.CrossEntropyLoss() # including softmax
    criterion_ats = nn.MSELoss()
    for epoch in args.epochs:
        print('Finetuning epoch {}/{}'.format(epoch, args.epochs))
        loss_cls_list = []
        loss_ats_list = []
        for i, (img, label) in tqdm(enumerate(loader)):
            if args.cuda:
                img = img.cuda()
            logits, activations_s = s_model(img)
            _, activations_t = t_model(img)
            loss_cls = criterion_cls(logits, label)
            loss_ats = criterion_ats(torch.stack(activations_s), torch.stack(activations_t))
            loss_cls_list.append(loss_cls.item())
            loss_ats_list.append(loss_ats.item())
            loss = loss_cls + args.beta * loss_ats
            if (i + 1) % 100 == 0:
                print('    loss: {}'.format(loss.item()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch average loss_cls: {}'.format(np.mean(loss_cls_list)))
        print('Epoch average loss_ats: {}'.format(np.mean(loss_ats_list)))


def run_test(args, model, loader):
    model.eval()
    preds = []
    labels = []
    for i, (img, label) in tqdm(enumerate(loader)):
        if args.cuda:
            img = img.cuda()
        logits, _ = model(img)
        preds += torch.argmax(logits, dim=1).tolist()
        labels += label.tolist()
    print('=====TEST RESULTS=====\n')
    print(classification_report(labels, preds, digits=4))


def main(args):
    transform = [transforms.ToTensor()]
    if args.norm:
        transform.append(transforms.Normalize((.5, .5, .5), (.5, .5, .5)))
    transform = transforms.Compose(transform)
    data_dir = args.data_dir + '/' + args.task
    if args.task == 'cat_dog':
        Loader = CatDogLoader
        num_classes = 2
    elif args.task == 'waste':
        Loader = WasteLoader
        num_classes = 2
    elif args.task == 'gtsrb':
        Loader = GTSRBLoader
        num_classes = 2
    else:
        raise NotImplementedError()

    train_loader = Loader(
        root=data_dir,
        batch_size=args.batch_size,
        split='train',
        transform=transform)

    test_loader = Loader(
        root=data_dir,
        batch_size=args.batch_size,
        split='test',
        transform=transform)

    student_model = get_model(args, 1000, args.pretrained_ckpt)
    teacher_model = get_model(args, num_classes, args.finetuned_ckpt)
    if args.cuda:
        student_model = student_model.cuda()
        teacher_model = teacher_model.cuda()

    run_train(args, student_model, teacher_model, train_loader)
    run_test(args, student_model, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyper-parameters settings
    parser.add_argument('--batch_size', default=256, type=int,
        help='Batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
        help='Learning rate')
    parser.add_argument('--epochs', default=10, type=int,
        help='Finetuning epochs')
    parser.add_argument('--beta', default=1, type=float,
        help='Activation beta')

    # parser.add_argument('--model', choices=['VGG', 'VGG_bn', 'vit'],
    #     help='Model choice')
    parser.add_argument('--data_dir', type=str,
        help='Path to dataset directory')
    parser.add_argument('--task', choices=['cat_dog', 'waste', 'gtsrb'],
        help='Task name')
    parser.add_argument('--norm', action='store_true', default=False,
        help='Enable normalization')
    
    # Load checkpoint
    parser.add_argument('--pretrained_ckpt', type=str,
        help='Pretrained model checkpoint')
    parser.add_argument('--finetuned_ckpt', type=str,
        help='Finetuned model checkpoint')
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print('Settings:')
    for arg in vars(args):
        print('\t{}: {}'.format(arg, getattr(args, arg)))
    main(args)