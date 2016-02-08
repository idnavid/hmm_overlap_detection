#! /usr/bin/python 

import numpy as np
import scipy.io.wavfile as wav
import pylab
import sys
import scikits.audiolab
import os 

sys.path.append('/scratch2/nxs113020/pyknograms/code/tools/pykno')
from pyknogram_extraction import *

def mix_files(f1,f2):

    base1 = f1.split('/')[-1].split('.wav')[0]
    base2 = f2.split('/')[-1].split('.wav')[0]

    (fs,sig) = wav.read(f1)
    s1 = sig.reshape((len(sig),1))
    del sig
    (fs,sig) = wav.read(f2)
    s2 = sig.reshape((len(sig),1))
    del sig
    block_length = 5*fs
    s1_blocks = enframe(s1,block_length,block_length)
    s2_blocks = enframe(s2,block_length,block_length)
    del s1, s2
     
    nrg1 = 0.707*np.sqrt(np.sum(np.power(s1_blocks,2),axis=1))
    nrg2 = 0.707*np.sqrt(np.sum(np.power(s2_blocks,2),axis=1))
     
    for i in range(len(nrg1)):
        db1 = np.log(nrg1[i])
        db2 = np.log(nrg2[i])
        if (db1 >= 9) and (db2 >= 9) and (0.1 < abs(db1 - db2) < 5):
            sir = '%.2f' % (db1 - db2)
            ovl_name = '/erasable/nxs113020/wav_ovl/'+base1+'_'+base2+'_sir'+sir+'_'+str(i)+'.wav'
            overlapped = s1_blocks[i,:] + s2_blocks[i,:]
            nrg_ovl = 0.707*np.sqrt(np.sum(np.power(overlapped,2)))
            scikits.audiolab.wavwrite(overlapped/nrg_ovl, ovl_name, fs, 'pcm16')
            


if __name__=='__main__':
    train_file = open('lists/train.txt')
    cwd = '/scratch/nxs113020/hmm_overlap_detection'
    train_list = []
    for i in train_file:
        train_list.append(i.strip())
    train_file.close()
    
    fjobs = open('lists/mix_jobs.txt','w')
    for i in range(len(train_list)):
        wav1 = train_list[i]
        base1 = wav1.split('Headset')[0]
        for j in range(i,len(train_list)):
            wav2 = train_list[j]
            base2 = wav2.split('Headset')[0]
            if not(wav1 == wav2) and (base1 == base2):
                # the following command allows us to directly use a function from this module in bash. 
                # Note: with this command, we don't need to have a separate bash script to use SGE. 
                fjobs.write(""". ~/.bashrc; python -c \"import sys; sys.path.append(\'%s\'); import mix_channels; mix_channels.mix_files(\'%s\', \'%s\')\" \n"""%(cwd,wav1,wav2))
    fjobs.close()
    os.system('/home/nxs113020/bin/myJsplit -M 100 -b 3 -q 1 -n mix_ami lists/mix_jobs.txt')

