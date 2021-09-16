#!/usr/bin/env python
# coding: utf-8

import subprocess as sub
import os
import time

# The following code is to create multiple containers in docker and input multiple files containing multiple candidate
# molecules into different containers. Tandem mass spectrometry of the candidate molecules are predicted and compared
# with the experimental tandem mass spectrometry data using CFM-ID4.

dkg=1
# dkg is the amount of change in the port of the host to the container, dk is the port of the host to the container

path=''
# 'path' is the path where the candidate molecule file is located, while mounting with the container as required

for i in range(1,11):
        dk = str(4000 + dkg)
        dkg += 1
        cmd2=':80 -v '+path+':/cfmid/public/ -i  wishartlab/cfmid:latest sh -c "cd /cfmid/public/' \
             ' && cfm-id sp.txt AN_ID test'+str(i)+'.txt -1 10 0.01 0.001 /trained_models_cfmid4.0/[M+H]+' \
                                                   '/param_output.log /trained_models_cfmid4.0/[M+H]+/param_config.txt Dice 1 out'+str(i)+'.txt '+str(i)+'.msp"'
        sub.Popen(cmd1+dk+cmd2,shell=True)

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


