# License: BSD
# Author: Sasank Chilamkurthy
# Initial Source: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# Adapted By: Seth Juarez (Microsoft)

from __future__ import print_function, division

import argparse
from pathlib import Path
import torch
import torch.onnx as onnx
import onnx as nx
from onnx import optimizer
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


from azureml.core.run import Run

plt.ioff()

###################################################################
# Helpers                                                         #
###################################################################
def info(msg, char = "#", width = 75):
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1*width)+5, msg) + char)
    print(char * width)

def check_dir(path, check=False):
    if check:
        assert os.path.exists(path), '{} does not exist!'.format(path)
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        return Path(path).resolve()

###################################################################
# Model Building                                                  #
###################################################################
def example_batch(run, inp, output_file, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
    # save sample batch
    if run != None:
        run.log_image('Example_Batch', plot=plt)
    else:
        plt.savefig(str(output_file), bbox_inches='tight')
    plt.close()

def visualize_model(run, model, dataloader, class_names, device, output_file, num_images=20, cols=5):
    was_training = model.training
    model.eval()
    images_so_far = 0

    rows = num_images//cols if num_images % cols == 0 else num_images//cols + 1
    _, axes = plt.subplots(rows, cols, figsize=(18, 7), 
                      subplot_kw={'xticks':[], 'yticks':[]},
                      gridspec_kw=dict(hspace=0.4, wspace=0.1))
    
    axs = enumerate(axes.flat)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                _, ax = next(axs)
                inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                inp = std * inp + mean
                inp = np.clip(inp, 0, 1)
                ax.imshow(inp)
                ax.set_title('prediction: {}'.format(class_names[preds[j]]))

                if images_so_far == num_images:
                    model.train(mode=was_training)

                    # save validation visualization
                    if run != None:
                        run.log_image('Validation_Run', plot=plt)
                    else:
                        plt.savefig(str(output_file), bbox_inches='tight')
                    plt.close()
                    return
        model.train(mode=was_training)
    

    
def get_data(data_dir='data', batch_size=4):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                batch_size=batch_size,
                                                shuffle=True, num_workers=4)
                    for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names

def train_model(model, device, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, run=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        info('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            batch = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                loss = loss.item()
                if phase == 'train':
                    if run != None:
                        run.log('{}_loss'.format(phase), loss)
                    print('	loss: {:>10f}  [{:>3d}/{:>3d}]'.format(loss, batch * len(inputs), dataset_sizes[phase]))

                running_loss += loss * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                batch += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if run != None:
                run.log('{}_epoch_accuracy'.format(phase), epoch_acc.item())
                run.log('{}_epoch_loss'.format(phase), epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                scheduler.step()
                print(f'Current lr: {scheduler.get_last_lr()}')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    if run != None:
        run.log('elapsed', time_elapsed)
        run.log('best_accuracy', best_acc)

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model

def start(data_dir='data', output_dir='outputs', batch_size=4, epochs=25, lr=0.001):
    
    # check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info('Initialization (using {})'.format(device))

    # AML Logging (if available)
    run = Run.get_context()
    if 'offlinerun' in run.id.lower():
        run = None
        print('Offline logging...')
    else:
        print('Using AML Logging...')
        run.log('data', data_dir)
        run.log('output', output_dir)
        run.log('epochs', epochs)
        run.log('batch', batch_size)
        run.log('learning_rate', lr)
        run.log('device', device)

    # get data loaders
    dataloaders, dataset_sizes, classes = get_data(data_dir=data_dir, 
                                                    batch_size=batch_size)

    examples = output_dir.joinpath('{}.png'.format('example_batch')).resolve()
    print('Saving example batch to {}'.format(str(examples)))
    inputs, clsses = next(iter(dataloaders['train']))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    example_batch(run, out, examples, title=[classes[x] for x in clsses])

    print('Classes: ', classes)
    print('Examples:')
    for k, v in dataset_sizes.items():
        print('	{} - {}'.format(k, v))

    print('Loading pretrained resnet18')
    # load pre-trained resnet18
    
    resnet = models.mobilenet_v2(pretrained=True)
    #num_ftrs = resnet.
    
    model = nn.Sequential(
        resnet,
        nn.ReLU(),
        nn.Linear(1000, len(classes)),
        nn.Softmax(dim=1)
    )
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=lr)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(model, device, criterion, optimizer_ft, exp_lr_scheduler, 
                            dataloaders, dataset_sizes, num_epochs=epochs, run=run)


    print('Visualizing validation run...')
    validation = output_dir.joinpath('{}.png'.format('validation')).resolve()
    print('Saving validation to {}'.format(str(validation)))
    visualize_model(run, model, dataloaders['val'], classes, device, validation)

    info('Saving')
    name = "model"
    onnx_file = output_dir.joinpath('{}.onnx'.format(name)).resolve()
    pth_file = output_dir.joinpath('{}.pth'.format(name)).resolve()
    
    # create dummy variable to traverse graph
    x = torch.randint(255, (1, 3, 224, 224), dtype=torch.float).to(device) / 255
    onnx.export(model, x, onnx_file)
    
    print('Saved onnx model to {}'.format(onnx_file))

    # saving PyTorch Model Dictionary
    torch.save(model.state_dict(), pth_file)
    print('Saved PyTorch Model to {}'.format(pth_file))

    if run != None:
        print('Offline logging...')
        run.upload_file("model.onnx", str(onnx_file))

        run.register_model(model_name='bork', 
                            model_path="model.onnx", 
                            model_framework="PyTorch", 
                            model_framework_version=torch.__version__, 
                            description="Tacos vs Burritos")

        print('Registered model!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hot Dog vs Pizza')
    parser.add_argument('-d', '--data', help='directory to training and test data', default='data')
    parser.add_argument('-o', '--output', help='output directory', default='outputs')
    
    parser.add_argument('-e', '--epochs', help='number of epochs', default=5, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=4, type=int)
    parser.add_argument('-l', '--lr', help='learning rate', default=0.001, type=float)


    args = parser.parse_args()

    # enforce folder locatations
    args.data = check_dir(args.data).resolve()
    args.outputs = check_dir(args.output).resolve()

    start(data_dir=args.data, output_dir=args.outputs, batch_size=args.batch, epochs=args.epochs, lr=args.lr)