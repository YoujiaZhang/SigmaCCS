#!/usr/bin/env python
# coding: utf-8



import subprocess as sub
import os
import time



# 下面的代码是在docker建立多个容器，将含有多个候选分子的多个文件分别输入到不同的容器中。使用CFMID4进行候选分子的二级质谱预测以及
# 与实验二级质谱数据的比对
# The following code is to create multiple containers in docker and input multiple files containing multiple candidate
# molecules into different containers. Tandem mass spectrometry of the candidate molecules are predicted and compared
# with the experimental tandem mass spectrometry data using CFM-ID4.

dkg=1
# dkg为宿主机与容器对接的端口的改变量，dk为宿主机与容器对接的端口
# dkg is the amount of change in the port of the host to the container, dk is the port of the host to the container
path=''
# path为候选分子文件所在的路径，同时按要求与容器进行挂载
# 'path' is the path where the candidate molecule file is located, while mounting with the container as required
for i in range(1,11):
        dk = str(4000 + dkg)
        dkg += 1
        cmd2=':80 -v '+path+':/cfmid/public/ -i  wishartlab/cfmid:latest sh -c "cd /cfmid/public/' \
             ' && cfm-id sp.txt AN_ID test'+str(i)+'.txt -1 10 0.01 0.001 /trained_models_cfmid4.0/[M+H]+' \
                                                   '/param_output.log /trained_models_cfmid4.0/[M+H]+/param_config.txt Dice 1 out'+str(i)+'.txt '+str(i)+'.msp"'
        sub.Popen(cmd1+dk+cmd2,shell=True)



# 此处代码为判断输出文件是否生成，从而让使用人员知道程序是否正在运行以及运行的进度
# This code detects if the output file is generated, so that the user knows if the program is running and its progress
k=1
while k <= 10:
    if os.path.isfile(path+str(k)+'.txt '):
        print(str(k) + ' is exist')
        k += 1
    elif not os.path.isfile(path+str(k)+'.txt'):
        print(str(k) + ' is not exist')
        time.sleep(3)
print('over')


