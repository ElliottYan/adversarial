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

from dataset import imageNet, Dataset

import pdb

# use pretrained mdoel
# model = torchvision.models.inception_v3(pretrained=True, transform_input=False)
# model = torchvision.models.inception_v3(pretrained=True, transform_input=False)

import torchvision.models as models

# resnet18 = models.resnet18(pretrained=True)
# alexnet = models.alexnet(pretrained=True)
# squeezenet = models.squeezenet1_0(pretrained=True)
# vgg16 = models.vgg16(pretrained=True)
# densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)

models = [inception, ]  # alexnet, resnet18, squeezenet, vgg16, densenet,
# model_str = {resnet18: "resnet18", alexnet: "alexnet", squeezenet: "squeezenet", vgg16: "vgg16", densenet: "densenet",
#              inception: "inception"}

model_str = {inception:"inception"}

img_size = 299
batch_size = 32
input_dir = "/home/Data/"
# input_dir = "/home/nisl/adversarial/data/"


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
def project_optimize(origin_x, x, last_grad, epsilon=3e-2, num_iter=1):
    below = origin_x - epsilon
    above = origin_x + epsilon

    # nn.utils.clip_grad_norm([x, ],epsilon)
    # optimizer = optim.Adam([x, ], lr=0.01)
    # optimizer.step()

    def fgsm(x, lr=0.3):
        sign = torch.sign(x.grad)
        x_tmp = x.clone()
        x_tmp += sign * lr
        return x_tmp.data

    def ifgsm(x_grad, num_iter, lr=0.3):
        sign = torch.sign(x_grad)
        x_tmp = x.clone()
        x_tmp += sign * lr / num_iter
        x_tmp = x_tmp.data
        x_tmp = clip(x_tmp, above, below)
        x_tmp = clip(x_tmp, torch.ones(x_tmp.shape).cuda(), torch.zeros(x_tmp.shape).cuda())

        return x_tmp

    # improvements for ifgsm
    def momentum(x, last_grad, lr=0.3, gamma = 0.7):
        return ifgsm(x.grad * lr +  gamma * ag.Variable(last_grad), num_iter, lr=lr)


    def clip(x, above, below):
        # use two mask to dsefine bounds
        # pdb.set_trace()
        mask_above = (above >= x).float()
        tmp = mask_above * x + (1 - mask_above) * above

        mask_below = (below <= tmp).float()
        tmp = mask_below * tmp + (1 - mask_below) * below
        return tmp

    # x = ifgsm(x, epsilon, n_iter)


    # pdb.set_trace()
    # optimizer = optim.Adam([x, ], lr=0.3)
    # optimizer.step()
    # x = x.data
    # x = clip(x, above, below)
    # x = clip(x, torch.ones(x.shape).cuda(), torch.zeros(x.shape).cuda())
    # r = ifgsm(x, num_iter)
    # x = x.data

    r = momentum(x, last_grad)

    return r


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
    transforms.CenterCrop(img_size),
    # normalize,
    transforms.ToTensor(),
    # LeNormalize(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

out_file_path = "/home/nisl/adversarial/results/"

dataset = imageNet(input_dir, transform=tf)
print("Whole size of dataset is %d" % dataset.__len__())
n_classes = dataset.n_classes
loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
loss_func = nn.MSELoss()

# for epoch in range(10):
# n_iters = range(10, 5)
n_iters = range(10, 5, -1)
# n_iters = range(1,5)
results = []

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

        old_pre = []
        new_pre = []
        target_pre = []

        for batch_idx, (input, target, true) in enumerate(loader):

            if batch_idx > 100:
                # pdb.set_trace()
                break
            print(batch_idx)
            # if batch_idx == 16:
            #     pdb.set_trace()
            old_in = input.clone().cuda()
            for ix in range(len(old_in)):
                im_orgin = unloader(old_in[ix].cpu())
                im_orgin.save('results/%d_origin.jpg' % ix, 'jpeg')

            if torch.cuda.is_available():
                input = input.cuda()

            batch_size = input.shape[0]

            input_var = ag.Variable(input, requires_grad=True)

            # pdb.set_trace()
            old_out = model(input_var)
            old_out_one_hot = old_out.max(1)[1]
            # print("Origin_classification results are %s"%str(old_out_one_hot))

            precision = accuracy(old_out_one_hot, true)
            old_pre.append(precision)
            print("Origin_classification precision are %s" % precision)

            #     defalut value from cleverhans
            epsilon = 1e-1
            # used for momentum
            last_grad = torch.zeros_like(input).cuda()
            for i in range(n_iter):
                # pdb.set_trace()
                input_var = ag.Variable(input, requires_grad=True)

                # use volatile=True means just inference.
                out = model(input_var)
                target_label = torch.zeros((batch_size, n_classes))

                # when using NLL loss, no need to transfer to one-hot

                one_hot(target, target_label)
                #                 target_label = target_label.float()
                # # label need to be long tensor
                # # and match the output variable
                # target_label = ag.Variable(target_label.long()).cuda()

                # loss function can only take two variables
                # target can only be 1-D tensor
                model.zero_grad()
                loss = loss_func(out, ag.Variable(target_label.cuda()).view(-1))
                loss.backward()

                # optimizer.step()
                input = project_optimize(old_in, input_var, last_grad, epsilon=epsilon, num_iter=n_iter)

            # use for print out image
            input_var = ag.Variable(input)

            # pdb.set_trace()
            new_out = model(input_var)
            new_out = new_out.max(1)[1]

            for ix in range(input_var.data.shape[0]):
                im = unloader(input_var.data[ix].cpu())
                im.save('results/%d.jpg'%ix, 'jpeg')
            # pdb.set_trace()

            #     pdb.set_trace()
            #     for i in range()
            # for i in range(batch_size):
            #     imshow(input_var[i].data)
            #     imshow(old_in[i])

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

            new_pre.append(false_precision)
            target_pre.append(precision)
            # pdb.set_trace()

        old_ave_pre = 0.0
        new_ave_pre = 0.0
        tar_ave_pre = 0.0
        n = len(old_pre)
        for i in range(n):
            old_ave_pre += old_pre[i]/n
            new_ave_pre += new_pre[i]/n
            tar_ave_pre += target_pre[i]/n

        pdb.set_trace()




        print("Results are :%f, %f, %f!"%(old_ave_pre, new_ave_pre, tar_ave_pre))


        # outs = np.concatenate(outs, axis=0)
        # adv_egs = np.concatenate(adv_egs, axis=0)
        # ins = np.concatenate(ins, axis=0)

        results.append((old_ave_pre, new_ave_pre, tar_ave_pre))

        # out_file_path + model_str[model] + "_" + str(n_iter) + ".npy"
        # np.save(out_file_path + model_str[model] + "_" + str(n_iter) + "_outs" + ".npy", outs)
        # np.save(out_file_path + model_str[model] + "_" + str(n_iter) + "_ins" + ".npy", ins)
        # np.save(out_file_path + model_str[model] + "_" + str(n_iter) + "_adv" + ".npy", adv_egs)
        # pdb.set_trace()
    for ix, item in enumerate(results):
        print("Results for item %d"%ix)
        print(item)

    # precision =

