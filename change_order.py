import numpy as np
import os
import re

# 改变图片的顺序函数，输入为文件地址，
def change_picure_order(filepath):
    list_pic = []
    path = os.listdir(filepath) #获得filepath中的文件名list
    os.chdir(filepath) # 改变当前目录为filepath
    # 对于list中的所有文件
    for i,dir in enumerate(path):
        # split把文件地址按照括号内为间隔分成2份（这里以.为间隔，.之前为[0],.之后为[1]）
        if dir.split('.')[0] != i and str(i) + '.jpg' not in dir :  # 如果前面那部分不等于i（也就是说如果第i个图片的名字不是i.jpg，比如第二幅图是3.jpg）
            os.rename(dir,str(i) + '.jpg') # 那就把这个图片的名字改成对应的位置（即把3.jpg改成1.jpg，因为第一幅图是从0.jpg开始的）
        list_pic.append(dir.split('.')[0]) # 整理成一个list
    return sort_string(list_pic)  # 整理好顺序后的list作为函数的输出


# def eachfile(filepath):
#     list_pic = []
#     pathdir = os.listdir(filepath)
#     dir_list = []
#     for alldir in pathdir:
#         dir_list.append(alldir)
#         if alldir != '.DS_Store':
#             picture = os.path.join(filepath,alldir)
#             list_pic.append(picture)
#     return list_pic,dir_list
#
# def get_type_p(each_type):
#     type_p = []
#     for i in range(len(each_type)):
#         picture = eachfile(each_type[i])
#         type_p.append(picture)
#     return type_p
#
def sort_string(lst):
    def reorder_numbers(s):  # Divide Numbers and letters
        re_digits = re.compile(r'(\d+)')
        pieces = re_digits.split(s)  # Cut into Numbers and non-numbers
        pieces[1::2] = map(int, pieces[1::2])  # To convert part of a number into an integer
        return pieces
    return sorted(lst, key = reorder_numbers)  # Sort the previous function as the key
#
#
# list_pic,dir_list = eachfile('train')
#
# picture_list = get_type_p(list_pic)
#
# list_pic = sort_string(list_pic)
#
# print(dir_list)


pic_list = change_picure_order('test/丽江古城')
print(pic_list)
