#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 将mnist源数据还原为灰度图片
import os
from PIL import Image
import struct
import numpy as np


def read_image(filename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()

    """
    打开文件,分别读取魔数,图片个数,以及行数和列数,在struct中使用了’>IIII’,也就是大端规则读取四个整形数
    如果要读取一个字节,则可以用’>B’（当然,这里用没用大端规则都是一样的,因此只有两个或两个以上的字节才有用）
    """
    magic, image_num, rows, columns = struct.unpack_from('>IIII', buf, index)  #对应pack_into()
    index += struct.calcsize('>IIII')

    for i in range(image_num):
        """
        先创建一张空白的图片,其中的’L’代表这张图片是灰度图,最后逐个像素读取,然后写进空白图片里,最后保存图片
        """
        image = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
        image = np.array(image, dtype='uint8')  # 这里注意Image对象的dtype是uint8,需要转换
        image = image.reshape(28, 28)
        image = Image.fromarray(image)  # 将二维数组转换为图像

        print('Saving ' + str(i) + ' image')
        image.save('test/' + str(i) + '.png')


def read_label(filename, saveFilename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()

    magic, label_num = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')
    labelArr = [0] * label_num

    for x in range(label_num):
        labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')

    save = open(saveFilename, 'w')
    save.write(','.join(map(lambda x: str(x), labelArr)))
    save.write('\n')
    save.close()
    print('save labels success')


if __name__ == '__main__':
    if not os.path.exists("test"):
        os.mkdir("test")
    read_image('t10k-images.idx3-ubyte')
    read_label('t10k-labels.idx1-ubyte', 'test/label.txt')
