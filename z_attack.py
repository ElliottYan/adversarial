import torch
import math

import torch.autograd as ag
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import numpy as np

import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms

import PIL
import matplotlib.pyplot as plt

from dataset import imageNet

import pdb

# use pretrained mdoel
# model = torchvision.models.inception_v3(pretrained=True, transform_input=False)

import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)

models = [inception, ] #alexnet, resnet18, squeezenet, vgg16, densenet,
model_str = {resnet18:"resnet18", alexnet:"alexnet", squeezenet:"squeezenet", vgg16:"vgg16", densenet:"densenet", inception:"inception"}



img_size = 299
batch_size = 32
input_dir = "/home/Data/"


class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """

    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor





def one_hot(ids, out_tensor):
    """
    ids: (list, ndarray) shape:[batch_size]
    out_tensor:FloatTensor shape:[batch_size, depth]
    """
    # if not isinstance(ids, (list, np.ndarray)):
    #     raise ValueError("ids must be 1-D list or array")

    ids = torch.LongTensor(ids).view(-1, 1)
    out_tensor.zero_()
    out_tensor.scatter_(dim=1, index=ids, value=1.0)
    # out_tensor.scatter_(1, ids, 1.0)


def accuracy(pred, target):
    bz = pred.shape[0]
    target = ag.Variable(target).view(-1)
    if torch.cuda.is_available():
        target = target.cuda()
    correct = pred.eq(target)
    acc = correct.sum().float() / bz
    return float(acc)


# we wanna make sure the deviation from x isn't to far.
def project_optimize(origin_x, x, epsilon=1e-2):
    below = origin_x - epsilon
    above = origin_x + epsilon

    # nn.utils.clip_grad_norm([x, ],epsilon)
    # optimizer = optim.Adam([x, ], lr=0.01)
    # optimizer.step()

    def fgsm(x, lr=0.3):
        sign = torch.sign(x.grad)
        x_tmp = x.clone()
        x_tmp += sign * lr * epsilon
        return x_tmp.data

    def ifgsm(x, epsilon, lr=0.3):
        sign = torch.sign(x.grad)
        x_tmp = x.clone()
        x_tmp += sign * lr * epsilon
        x_tmp = clip(x, above, below)
        return x_tmp

    def clip(x, above, below):
        # use two mask to dsefine bounds
        x_data = x.data
        mask_above = (above >= x_data).float()
        tmp = mask_above * x_data + (1 - mask_above) * above

        mask_below = (below <= tmp).float()
        tmp = mask_below * tmp + (1 - mask_below) * below
        return tmp

    # x = ifgsm(x, epsilon)
    optimizer = optim.Adam([x,], lr = 0.03)
    optimizer.step()
    x = clip(x, above, below)

    return x


unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()


def imshow(tensor, title=None):
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(3, img_size, img_size)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

tf = transforms.Compose([
    #     transforms.Scale(int(math.floor(img_size / 0.875))),
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    # normalize,
    transforms.ToTensor(),
        LeNormalize(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

out_file_path = "/home/nisl/adversarial/results/"

dataset = torchvision.datasets.ImageFolder(input_dir, transform=tf)
print("Whole size of dataset is %d"%dataset.__len__())
#n_classes = dataset.n_classes
loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
loss_func = nn.NLLLoss()

# for epoch in range(10):
n_iters = range(1,5)



for model in models:

    if torch.cuda.is_available():
        model = model.cuda()

    # make model in evaluation mode.
    # unable some ops like dropout or BatchNorm turns off.
    model.eval()

    for n_iter in n_iters:
        print('Now is in ' + model_str[model] + ', iteration:' + str(n_iter))

        outs = []
        adv_egs = []
        ins = []

        for batch_idx, (input, target, true) in enumerate(loader):
            print(batch_idx)
            old_in = input

            if torch.cuda.is_available():
                input = input.cuda()

            batch_size = input.shape[0]

            input_var = ag.Variable(input, requires_grad=True)

            old_out = model(input_var)
            old_out = old_out.max(1)[1]
            # print("Origin_classification results are %s"%str(old_out))

            precision = accuracy(old_out, true)
            print("Origin_classification precision are %s" % precision)

            count = 0
            #     defalut value from cleverhans
            epsilon = 0.5e-2

            origin_x = input

            for i in range(n_iter):
                input_var = ag.Variable(input, requires_grad=True)

                # use volatile=True means just inference.
                out = model(input_var)
                target_label = torch.zeros((batch_size, n_classes))

                # when using NLL loss, no need to transfer to one-hot

                # one_hot(target, target_label)
                # # label need to be long tensor
                # # and match the output variable
                # target_label = ag.Variable(target_label.long()).cuda()

                # loss function can only take two variables
                # target can only be 1-D tensor
                model.zero_grad()
                loss = loss_func(out, ag.Variable(target.cuda()).view(-1))
                loss.backward()

                # optimizer.step()
                input = project_optimize(origin_x, input_var, epsilon=epsilon)

            # use for print out image
            input_var = ag.Variable(input)

            new_out = model(input_var)
            new_out = new_out.max(1)[1]

            #     pdb.set_trace()
            #     for i in range()
            #     for i in range(batch_size):
            #         imshow(input_var[i].data)
            #         imshow(old_in[i])

            # print("After optimize, the classification results are %s"%str(new_out))
            precision = accuracy(new_out, true)
            false_precision = accuracy(new_out, target)
            print('Target precision after attacking is %s' % false_precision)
            print('Precision after attacking is %s' % precision)


            # out = out.max(1)[1] + 1  # argmax + offset to match Google's Tensorflow + Inception 1001 class ids

            # include the adv_egs in outs
            outs.append(new_out.data.cpu().numpy())
            adv_egs.append(input_var.data.cpu().numpy())
            ins.append(old_in.cpu().numpy())

        outs = np.concatenate(outs, axis=0)
        adv_egs = np.concatenate(adv_egs, axis=0)
        ins = np.concatenate(ins, axis=0)

        out_file_path + model_str[model] + "_" + str(n_iter) + ".npy"
        np.save(out_file_path + model_str[model] + "_" + str(n_iter) + "_outs" +".npy", outs)
        np.save(out_file_path + model_str[model] + "_" + str(n_iter) + "_ins" + ".npy", ins)
        np.save(out_file_path + model_str[model] + "_" + str(n_iter) + "_adv" + ".npy", adv_egs)
        # pdb.set_trace()

    # precision =


