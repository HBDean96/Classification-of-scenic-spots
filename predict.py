import torch
from torchvision import datasets, models, transforms
import os
import warnings
from PIL import Image

warnings.filterwarnings('ignore')



data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.CenterCrop(224),
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

final_picture_path =  'judge_picture'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val','test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val','test']}


# final_image_datasets = {'test': datasets.ImageFolder(final_picture_path,
#                                           data_transforms['test'])}
#
#
# final_dataloaders = {'test': torch.utils.data.DataLoader(final_image_datasets['test'], batch_size=4,
#                                               shuffle=True, num_workers=4)}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}

class_names = image_datasets['train'].classes

class_names_test = image_datasets['test'].classes

device = "cuda:0" if torch.cuda.is_available() else "cpu"




# if __name__ == '__main__':
#     model = torch.load('scenery.pth')
#
#     eval_loss = 0.
#     eval_acc = 0.
#     s = 0.
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['train']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             for j in range(inputs.size()[0]):
#                 # s =s + int(class_names[preds[j]])
#                 # print(class_names[preds[j]])
#                 # if int(class_names[preds[j]]) == int(labels[j]):
#                 if class_names[preds[j]] == class_names[int(labels[j])]:
#                     s = s + 1
#     print(s)
#     print(s / (len(dataloaders['train']) * 4))
#
#     s = 0.
#     label_ = []
#     preds_list = []
#     with torch.no_grad():
#         for i, (inputs_test, labels_test) in enumerate(dataloaders['test']):
#             inputs_test = inputs_test.to(device)
#             labels_test = labels_test.to(device)
#             label_.append(labels_test)
#             outputs_test = model(inputs_test)
#             test_, preds_test = torch.max(outputs_test, 1)
#             preds_list.append(preds_test)
#             for k in range(inputs_test.size()[0]):
#                 # s =s + int(class_names[preds[j]])
#                 # print(class_names[preds[j]])
#                 # if int(class_names[preds[j]]) == int(labels[j]):
#                 if class_names[preds_test[k]] == class_names_test[int(labels_test[k])]:
#                     s = s + 1
#
#     print(s)
#     print(s / (len(dataloaders['test']) * 4))
#
#     print(preds_list)
#     labels_list = []
#     for preds_ in preds_list:
#         for label in preds_:
#             labels_list.append(class_names[label])

    # print(label_)
    # print(labels_list)



def predict_pic(picture_path):

    model = torch.load('scenery.pth')
    model.eval()


    image = Image.open(picture_path)

    # model.eval()
    image = data_transforms['test'](image)
    image = image.unsqueeze(0)
    inputs = image.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    predict = class_names[preds]

    return predict

if __name__ == '__main__':

    # picture_path = 'judge_picture/200.jpg'
    #
    # predict = predict_pic(picture_path)

    # print(predict)

    model = torch.load('scenery.pth')
    model.eval()

    a = 0.
    label_train = []
    preds_list_train = []
    with torch.no_grad():
        for i, (inputs_train, labels_train) in enumerate(dataloaders['train']):
            inputs_test = inputs_train.to(device)
            labels_train = labels_train.to(device)
            label_train.append(labels_train)
            outputs_train = model(inputs_train)
            train_, preds_train = torch.max(outputs_train, 1)
            preds_list_train.append(preds_train)
            for k in range(inputs_train.size()[0]):
                # s =s + int(class_names[preds[j]])
                # print(class_names[preds[j]])
                # if int(class_names[preds[j]]) == int(labels[j]):
                if class_names[preds_train[k]] == class_names[int(labels_train[k])]:
                    a = a + 1

    print('Train correct number: %d'%a)
    print('Train accuracy: %f' %(a / (len(dataloaders['train']) * 4)))



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

    print('Test correct number: %d'%s)
    print('Test accuracy: %f' %(s / (len(dataloaders['test']) * 4)))





    # print(preds_list)
    labels_list = []
    for preds_ in preds_list:
        for label in preds_:
            labels_list.append(class_names[label])

    # print(label_)
    # print(labels_list)
    # for label0,pre_label in zip(labels_,labels_list):
    #     print('label:%f,pre label:%f'%(label0,pre_label))
