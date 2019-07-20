import cv2 as cv
import numpy as np
import glob
from matplotlib import pyplot as plt
import torch
import torchvision
from torchvision import datasets, models, transforms
import os
import time
import copy
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import face_recognition
import warnings
warnings.filterwarnings('ignore')
# 参数设置
lr = 0.001  # 步长
epoch = 20  # 运行多少个epoch


# pytorch的数据转换器，可进行argumentation（数据增强）等操作。
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),  # 将图片调整至224*224的大小（因为densenet169要求输入的image大小为224*224）
        transforms.RandomCrop(224),  # 进行裁剪
        transforms.RandomHorizontalFlip(),  # 随机图像水平翻转 （概率为0.5）
        transforms.ToTensor(),  # 转换成pytorch所需要的Tensor格式
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # 进行图片的标准化操作（先将输入归一化到（0，1），再使用公式”(x-mean)/std”，将每个元素分布到(-1,1) ）
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256) ,
        transforms.RandomCrop(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}



data_dir = ''   #文件夹名称
#  创造图片的datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val','test']}
#  将图片的dataset输入进dataloader，以便等会从dataloader中输出transform之后的图像
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val','test']}
# 读取dataset的大小
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
# 获得训练集所有的图片类名（2000个图片，类名为10个景点的名字），和训练集的图片是一一对应的。
class_names = image_datasets['train'].classes
# 获得测试集的类名
class_names_test = image_datasets['test'].classes
# 设置为GPU模式
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 展示图片，这里可以不使用
def imshow(inp, title=None):
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



# 训练模型，输入为模型（densenet169），判断标准（交叉熵），优化器（SGD），调整学习率的方法（每7个epoch变成原来的0.1倍），和epoch.
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    # 复制原模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    # 初始化准确率
    best_acc = 0.0
    # 开始训练
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每个epoch都有一个培训和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # 将模型设置为训练模式
            else:
                model.eval()  # 设置模型为评估模式
            # 初始化loss和corrects
            running_loss = 0.0
            running_corrects = 0

            # 遍历所有数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 将梯度设置为0
                optimizer.zero_grad()

                # 前向传播
                # 如果进行训练将跟踪训练的过程
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 如果在训练阶段，进行反向传播和参数优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计loss和corrects
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # 计算每个epoch的loss和acc（准确率）
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            # 输出结果
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 深度复制模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

# 模型可视化 展示验证集的结果（这里没有使用），后面展示了训练集和测试集的，代码类似
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # print (outputs)
            print(labels)

            for j in range(inputs.size()[0]):
                # print (class_names[preds[1]])
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {},val:{}'.format(class_names[preds[j]], class_names[int(labels[j])]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if __name__ == '__main__':
    # 获取一批训练数据
    inputs, classes = next(iter(dataloaders['train']))

    # 从批处理中生成gird（一批图像拼成一副图像）
    out = torchvision.utils.make_grid(inputs)
    # 展示出来
    imshow(out, title=[class_names[x] for x in classes])

    model_ft = models.densenet169(pretrained=True)  # 这一句表示加载densnet169在imagnet数据集上的预训练模型，True表示不用重新下载，false会自动下载模型（需要翻墙）
    # 特征数量
    num_ftrs = model_ft.classifier.in_features

    model_ft.classifier = nn.Linear(num_ftrs, 44)  # 这两句重新拟合模型分类
    # 并行处理使用GPU
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    # 将criterion设置为交叉熵
    criterion = nn.CrossEntropyLoss()

    # 观察所有参数都在被优化，使用的是带momentum的SGD（带动量的随机梯度下降），momentum设置为0.9，lr在最前面调整。
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)

    # 每7个epoch将lr调整为原来的0.1倍
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # 调用训练函数训练模型
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=epoch)

    visualize_model(model_ft)
    # 保存整个模型
    torch.save(model_ft, 'scenery.pth')
    # torch.save(demo.state_dict(), 'people.pth')
    # 整体精度
    # show acc
    # 读取模型
    model = torch.load('scenery.pth')
    # 将loss和acc设置为0
    eval_loss = 0.
    eval_acc = 0.
    s = 0.
    with torch.no_grad():    # 没有梯度更新
        # 从dataloaders中加载出图像和对应的labels
        for i, (inputs, labels) in enumerate(dataloaders['train']):
            # 调用GPU处理
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 喂进神经网络
            outputs = model(inputs)
            # 得到预测标签序列
            _, preds = torch.max(outputs, 1)
            # 将得到的预测标签序列与原始训练集的类别序列比较，一样则+1，不一样则跳过
            for j in range(inputs.size()[0]):
                # s =s + int(class_names[preds[j]])
                # print(class_names[preds[j]])
                # if int(class_names[preds[j]]) == int(labels[j]):
                if class_names[preds[j]] == class_names[int(labels[j])]:
                    s = s + 1
    # s为训练集与预测数据相等的个数
    print(s)
    # s除以总个数，得到准确率（即有多少张图片被预测为正确的类别）
    print(s / (len(dataloaders['train']) * 4))

    # 下面是测试集的预测过程与训练集基本相同
    s = 0.
    label_ = []
    preds_list = []
    with torch.no_grad():
        for i, (inputs_test, labels_test) in enumerate(dataloaders['test']):
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)
            label_.append(labels_test)
            outputs_test = model(inputs_test)
            test_, preds_test = torch.max(outputs_test, 1)
            preds_list.append(preds_test)
            for k in range(inputs_test.size()[0]):
                # s =s + int(class_names[preds[j]])
                # print(class_names[preds[j]])
                # if int(class_names[preds[j]]) == int(labels[j]):
                if class_names[preds_test[k]] == class_names_test[int(labels_test[k])]:
                    s = s + 1

    print(s)
    print(s / (len(dataloaders['test']) * 4))


