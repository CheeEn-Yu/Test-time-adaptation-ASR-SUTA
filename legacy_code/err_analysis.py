import cn2an
from opencc import OpenCC
import nltk
from nltk import ngrams
cc = OpenCC('t2s')

import csv
from jiwer import wer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from suta import *
# exp_name = f'./ex_data/suta_whisper_pratio0801'
exp_name = f'./ex_data/suta_orip_allEncoder0803'

import re
import numpy as np
# 開啟文件
with open(f'{exp_name}/result.txt', 'r') as file:
    # 讀取文件所有行
    lines = file.readlines()

# 初始化兩個列表
ori_wers = []
wer_1 = []
wer_3 = []
wer_5 = []
wer_10 = []
wer_13 = []
wer_15 = []
wer_18 = []
wer_20 = []
wer_24 = []
wer_27 = []
wer_30 = []
wer_33 = []
wer_36 = []
wer_39 = []
ori_transcription = []
transcriptions_1 = []
transcriptions_3 = []
transcriptions_5 = []
transcriptions_10 = []
transcriptions_13 = []
transcriptions_15 = []
transcriptions_18 = []
transcriptions_20 = []
labels = []
# 遍歷每一行
for line in lines:
    # if len(labels) == 20:
    #     continue
    # if len(labels) == 50:
    #     break
    # 刪除每行開頭和結尾的空白字符
    line = line.strip()
    # 使用":"分割該行，取第二部分作為句子
    try:
        sentence = line.split(':')[1]
        match = re.search(r'\((.*?)\)', (line.split(':')[0]))
        ori_value = float(match.group(1))
    except:
        continue
    # 如果該行以"ori"開頭
    if line.startswith('ori'):
        # 使用":"分割該行，取第二部分作為ori的值
        ori_wer = float(match.group(1))
        # 將ori的值添加到ori_wers列表中
        ori_wers.append(ori_wer)
        ori_transcription.append(sentence)
    # 如果該行以"label"開頭
    elif line.startswith('label'):
        labels.append(line.split(':')[1])
    elif line.startswith('step0'):
        ori_value = float(match.group(1))
        # 將ori的值添加到ori_values列表中
        wer_1.append(ori_value)
        sentence = line.split(':')[1]
        transcriptions_1.append(sentence)
    elif line.startswith('step30'):
        ori_value = float(match.group(1))
        wer_30.append(ori_value)
        print(line)
    elif line.startswith('step33'):
        ori_value = float(match.group(1))
        wer_33.append(ori_value)
    elif line.startswith('step36'):
        ori_value = float(match.group(1))
        wer_36.append(ori_value)
    elif line.startswith('step39'):
        ori_value = float(match.group(1))
        wer_39.append(ori_value)
    elif line.startswith('step3'):
        ori_value = float(match.group(1))
        # 將ori的值添加到ori_values列表中
        wer_3.append(ori_value)
        sentence = line.split(':')[1]
        transcriptions_3.append(sentence)
    elif line.startswith('step6'):
        ori_value = float(match.group(1))
        # 將ori的值添加到ori_values列表中
        wer_5.append(ori_value)
        sentence = line.split(':')[1]
        transcriptions_5.append(sentence)
    elif line.startswith('step9'):
        ori_value = float(match.group(1))
        # 將ori的值添加到ori_values列表中
        wer_10.append(ori_value)
        sentence = line.split(':')[1]
        transcriptions_10.append(sentence)
    elif line.startswith('step12'):
        ori_value = float(match.group(1))
        # 將ori的值添加到ori_values列表中
        wer_13.append(ori_value)
        sentence = line.split(':')[1]
        transcriptions_13.append(sentence)
    elif line.startswith('step14'):
        ori_value = float(match.group(1))
        # 將ori的值添加到ori_values列表中
        wer_15.append(ori_value)
        sentence = line.split(':')[1]
        transcriptions_15.append(sentence)
    elif line.startswith('step18'):
        ori_value = float(match.group(1))
        # 將ori的值添加到ori_values列表中
        wer_18.append(ori_value)
        sentence = line.split(':')[1]
        transcriptions_18.append(sentence)
    elif line.startswith('step21'):
        ori_value = float(match.group(1))
        wer_20.append(ori_value)
        sentence = line.split(':')[1]
        transcriptions_20.append(sentence)
    elif line.startswith('step24'):
        ori_value = float(match.group(1))
        wer_24.append(ori_value)
    elif line.startswith('step27'):
        ori_value = float(match.group(1))
        wer_27.append(ori_value)

try:
    print(f'{np.array(ori_wers).mean() * 100}')
    print(f'{np.array(wer_1).mean() * 100}')
    print(f'{np.array(wer_3).mean() * 100}')
    print(f'{np.array(wer_5).mean() * 100}')
    print(f'{np.array(wer_10).mean() * 100}')
    print(f'{np.array(wer_13).mean() * 100}')
    print(f'{np.array(wer_15).mean() * 100}')
    print(f'{np.array(wer_18).mean() * 100}')
    print(f'{np.array(wer_20).mean() * 100}')
except:
    pass
best_wers = [min(wer_10[i],wer_13[i],wer_15[i]) for i in range(len(wer_15))]
worst_wers = [max(wer_10[i],wer_13[i],wer_15[i]) for i in range(len(wer_15))]
# best_wers = [ min(ori_wers[i], wer_1[i], wer_3[i],wer_5[i],wer_10[i]) for i in range(len(wer_1))]
# best_wers = [ min(wer_1[i], wer_3[i],wer_5[i],wer_10[i]) for i in range(len(wer_1))]

# elem 0 means better, elem 1 means worse, elem 2 means no change
best_count = [0, 0, 0]
worst_count = [0, 0, 0]
for i in range(len(wer_15)):
    if best_wers[i] < ori_wers[i]:
        best_count[0] +=1
    elif best_wers[i] > ori_wers[i]:
        best_count[1] +=1
    elif best_wers[i] == ori_wers[i]:
        best_count[2] +=1
    if worst_wers[i] < ori_wers[i]:
        worst_count[0] +=1
    elif worst_wers[i] > ori_wers[i]:
        worst_count[1] +=1
    elif worst_wers[i] == ori_wers[i]:
        worst_count[2] +=1
print('min lower than ori:')
print(best_count)
print('max lower than ori:')
print(worst_count)
