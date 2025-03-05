#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 17:33:16 2017
@author: chirag212
@author: nivi_k (added groups)

"""

#==============================================================================
# Data Pre-Processing 
# Pitt corpus -- DementiaBank -- Cookie theft description task
# All patient-investigator discourses were transcribed using CHAT protocol
#==============================================================================

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def process_string(content):    
    # Processing strings
    string = """<|im_start|>system In the following dialogue, you will play the role of a doctor conducting an assessment of a patient's cognitive and linguistic abilities through the description of a picture known as the "Cookie Theft" scene. This is part of an early screening process for Alzheimer's Disease (AD). The picture depicts a scene in a kitchen: a mother is washing dishes at the sink, unaware that the faucet is running and causing water to overflow. Meanwhile, two children are taking advantage of the situation to steal cookies from a jar. One boy is climbing a wobbly stool to reach the cookies, while a girl stands by, ready to catch them. Your task is to guide the patient, referred to as PAR, to describe this scene in detail, including the characters, actions, and background details. Pay attention to PAR's language expression, information management, vocabulary usage, referential coherence, psychological state language, and grammar and pronunciation. Throughout the dialogue, maintain patience and professionalism, using a gentle and encouraging tone to guide PAR and ensure they can express their thoughts fully. You will be referred to as INV. Now, begin the conversation with PAR.<|im_end|>"""

    """
    <|im_start|>system
    在接下来的对话中，你将扮演一位医生，通过引导患者描述一幅名为“饼干偷图”的图片，来评估患者的认知和语言能力，这是阿尔茨海默病（AD）早期筛查的一部分。这幅图片描绘了厨房中的一幕：一位母亲在水槽旁洗碗，没有注意到水龙头开着导致水溢出。与此同时，两个孩子趁机从罐子里偷饼干。其中一个男孩爬上摇摇欲坠的凳子去拿饼干，另一个女孩站在旁边，伸手准备接饼干。你的任务是引导患者（称为PAR）详细描述这个场景，包括人物、动作和背景细节。注意观察PAR的语言表达、信息管理、词汇使用、指称连贯性、心理状态语言以及语法和发音。在整个对话过程中，保持耐心和专业，用温和、鼓励性的语气引导PAR，确保他们能够充分表达自己的想法。你将被称为INV。现在，开始与PAR的对话。
    <|im_end|>
    """

    flag = 0
    for f in content:
        # if content[6].split(':')[1].split('|')[5] == 'Control':
        #     break;
        # if f != "*PAR:	and there's a &+s stool that he is on and it already is starting to":
        #     continue
        # if f.startswith('%mor') or f.startswith('*INV'): // 自己动手debug
        # '<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>\n'
        # 提示词 用户 响应
        # 提示词 医生 患者

        if f.startswith('%wor') or f.startswith('%mor') or f.startswith('%gra') or f.startswith('*INV'):
            flag = 0
        if f.startswith('*PAR:'):
            flag = 1
            string = string + "<|im_start|>PAR<PAR>: " + f[5:]   # <INV>: OR <PAR>:
        if f.startswith('*INV:'):
            flag = 1
            string = string + "<|im_start|>INV<INV>: " + f[5:]
#        print (string)
            
    string = ''.join(i for i in string if not i.isdigit())

    # COUNT FILLERS ================================================================

    # Count of trailing
    count_trailing = string.count('+...')
    
    # Count pauses
    count_pause = []
    count_1 = string.count("(.)")
    string = string.replace('(.)', '')
    count_2 = string.count("(..)")
    string = string.replace('(..)', '')
    count_3 = string.count("(...)")
    string = string.replace('(...)', '')
    count_pause = [count_1, count_2, count_3]
    
    # Count of unintelligible words
    count_unintelligible = string.count('xxx')
    
    # Count of repetitions
    count_repetitions = string.count('[/]')
    
    count_misc = [count_unintelligible, count_trailing, count_repetitions]
    
    # REMOVE EXTRA TAGS and FILLERS
    
    string = string.replace("\t", ' ')
    #==============================================================================
    #     Group 1
    #==============================================================================
    # Remove paranthesis '()'
    string = string.replace('(', '')
    string = string.replace(')', '')

    #==============================================================================
    #     Group 2
    #==============================================================================
    # Remove paranthesis '&=clears throat'
    string = string.replace('&=clears throat', '')
    string = string.replace('&=clears:throat', '')
	    
    # Remove paranthesis '&=anything', '&anything', and '=anything'
    bad_chars = ["&=", '&', "=", "+"]
    
    for bad_str in bad_chars:
        string = ' '.join(s for s in string.split(' ') if not s.startswith(bad_str))
    
    #==============================================================================
    #     Group 3
    #==============================================================================
    # Remove paranthesis '[* anything]', '[+ anything]', [: anything], '[=anything]',
    #                    '[/anything]', '[x anything]', and'[% anything]'
    bad_chars = ["[*", "[+", "[:", "[=", "[/", "[x", "[%"]
    for bad_str in bad_chars:
        string = ' '.join(s for s in string.split(' ') if not s.startswith(bad_str) and not s.endswith(']'))
    
    #string = re.sub(r'\[.*\]', '', string)
      
    #==============================================================================
    #     Group 4, 5, 6, 7
    #==============================================================================
    bad_chars = ["+=", '<', '>', '^', "xxx", '@', " _ ", " _ :", "+//"]
    for bad_str in bad_chars:
        string = string.replace(bad_str, '')

    #==============================================================================
    #     Group 8
    #==============================================================================        
    bad_chars = ["sr-ret", "pw-ret", "sr-rep", "s:r-ret", "p:w-ret", "s:r-rep", "s:r"]
    for bad_str in bad_chars:
        string = string.replace(bad_str, '')        
    
    #==============================================================================
    #     Group 9
    #==============================================================================
    bad_chars = ["[", "]", ":","_","-","+", "*", '\x15']
    for bad_str in bad_chars:
        string = string.replace(bad_str, '')    

	#==============================================================================
    #     Group 10
    #==============================================================================
    bad_chars = ["mhm .", "hmhmm .", "hmm .", "okay .", "hm .", "alright .", "well .", "oh ."]
    for bad_str in bad_chars:
        string = string.replace(bad_str, '') 

		
    string = string.replace('  ', ' ')
    string = string.replace('..', '.')
    string = string.replace('. .', '.')
    return string, count_pause, count_misc

    
def main():
    parser = argparse.ArgumentParser(description='Processing Dementia data')
    
    parser.add_argument('--file_path', default=r'C:\Users\lxq717machine\Desktop\DementiaBank\Pitt', type=str,
                        help='filepath for Control and Dementia folders')
#    parser.add_argument('--lr', default=0.0663, type=int,
#                        help='learning rate')
#    parser.add_argument('--epochs', default=100, type=int,
#                        help='number of epochs')

    args = parser.parse_args()
    p_id_control = []
    p_id_dementia = []
    # Filenames
    control_path = os.path.join(args.file_path, 'Control', 'cookie')
    dementia_path = os.path.join(args.file_path, 'Dementia', 'cookie')
    control_list = os.listdir(control_path)
    dementia_list = os.listdir(dementia_path)

    # MetaData
    idx = 0
    temp = []
    index = range(len(control_list) + len(dementia_list))
    columns = ['filepath', 'age', 'gender', 'mmse', 'pause1','pause2', 'pause3', 
               'count_unintelligible', 'count_trailing', 'count_repetitions',  
               'category', 'data']
    
    metadata = pd.DataFrame(index=index, columns=columns)
    for file in control_list:
        
         with open(os.path.join(control_path, file),encoding="utf8") as f:
             content = f.read().splitlines()
         
         p_id_control.append(os.path.join(control_path, file).split('/')[-1].split('-')[0])
         dialogue, count_pause, count_misc = process_string(content)
         
         for s in dialogue.split(' '):
             if s.startswith("&"):
                 temp.append(s)
                 
         age = content[6].split(':')[1].split('|')[3][:-1]
         gender = content[6].split(':')[1].split('|')[4]
         MMSE = content[6].split(':')[1].split('|')[8]
         category = content[6].split(':')[1].split('|')[5]
         
#         Uncomment the following 'if' statement to save only the files with an assigned MMSE score
#         if MMSE != "":
         metadata.loc[idx] = [file, age, gender, \
                             MMSE, count_pause[0], count_pause[1], count_pause[2], 
                             count_misc[0], count_misc[1], 
                             count_misc[2], category, dialogue]
         idx += 1
             
    for file in dementia_list:
         with open(os.path.join(dementia_path, file),encoding="utf8") as f:
             content = f.read().splitlines()
         
         p_id_dementia.append(os.path.join(dementia_path, file).split('/')[-1].split('-')[0])
         dialogue, count_pause, count_misc = process_string(content)

         for s in dialogue.split(' '):
             if s.startswith("&"):
                 temp.append(s)
                 
         age = content[6].split(':')[1].split('|')[3][:-1]
         gender = content[6].split(':')[1].split('|')[4]
         MMSE = content[6].split(':')[1].split('|')[8]
         category = content[6].split(':')[1].split('|')[5]
         
#         Uncomment the following 'if' statement to save only the files with an assigned MMSE score           
#         if MMSE != "":
         metadata.loc[idx] = [file, age, gender, \
                             MMSE,  count_pause[0], count_pause[1], count_pause[2], 
                             count_misc[0], count_misc[1], 
                             count_misc[2], category, dialogue]
         idx += 1
         
# =============================================================================
#    View category distribution         
# =============================================================================
#    categories = np.unique(metadata['category'])
#    count_category = [np.count_nonzero(metadata['category']==cat) for cat in categories]    
#    categories = tuple(categories)
#    y_pos = np.arange(len(count_category))
#    sort_label = np.argsort(count_category)
#    temp_data = np.array([[categories[i], count_category[i]] for i in sort_label])    
#    plt.bar(y_pos, temp_data[:, 1], align='center', alpha=0.8)
#    plt.xticks(y_pos, temp_data[:, 0], fontsize=21.0, fontweight='bold', rotation='vertical')
#    plt.ylabel('Categorical count', fontsize=21.0, fontweight='bold')
#    plt.yticks(fontsize=21.0, fontweight='bold')
#    plt.tight_layout()
#    plt.show()
# =============================================================================
    
    metadata.to_csv(os.path.join(args.file_path, 'metadata_allvisits.csv'), index=False, encoding='utf-8')   
#    metadata.to_csv(os.path.join(args.file_path, 'metadata_MMSEvisits.csv'), index=False, encoding='utf-8')   
    return np.unique(temp), p_id_control, p_id_dementia

if __name__ == '__main__':
    x, p_id_control, p_id_dementia = main()