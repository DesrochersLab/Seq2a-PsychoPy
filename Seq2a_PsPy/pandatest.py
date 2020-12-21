# -*- coding: utf-8 -*-
"""
  fMRI toy data tests; 09/18/20
"""
import pandas as pd
#import numpy as np
import collections.abc

xls = 'Seq2a_test_noblanks.xls'
trial_data = pd.read_excel(xls, ['Sheet1','Sheet2','Sheet3','Sheet4'], index_row=0)

S1,S2,S3,S4 = trial_data['Sheet1'],trial_data['Sheet2'],trial_data['Sheet3'],trial_data['Sheet4']

S1.to_csv(path_or_buf='S1.csv',index=False)
S2.to_csv(path_or_buf='S2.csv',index=False)
S3.to_csv(path_or_buf='S3.csv',index=False)
S4.to_csv(path_or_buf='S4.csv',index=False)

tr_num=0

def TaskPromptToggle(task_id):
    if str(task_id) == '1': colorYN, shapeYN = 1, 0
    else                  : colorYN, shapeYN = 0, 1
    return( colorYN, shapeYN )
# Value Conversions; MATLAB idx to rgb, verticies, size
# The MATLAB script handles our variables as 1's and 2's, but we need to do some
# translating before psychopy can use our stimuli. we'll do this by using the 
# stimulus ids, and converting the unique color/shape/size params to something
# psychopy expects to see. Key for below: stim_idx:(color,shape,size).
# 1:(1,1,1) 2:(1,1,2) 3:(1,2,1) 4:(1,2,2)
# 5:(2,1,1) 6:(2,1,2) 7:(2,2,1) 8:(2,2,2)
def StimByIdx(stimulus_id):
    if   stimulus_id == 1: color_array, shape_array, size_array = [1,0,0], 40, 1
    elif stimulus_id == 2: color_array, shape_array, size_array = [1,0,0], 40, 2
    elif stimulus_id == 3: color_array, shape_array, size_array = [1,0,0],  4, 1
    elif stimulus_id == 4: color_array, shape_array, size_array = [1,0,0],  4, 2
    elif stimulus_id == 5: color_array, shape_array, size_array = [0,0,1], 40, 1
    elif stimulus_id == 6: color_array, shape_array, size_array = [0,0,1], 40, 2
    elif stimulus_id == 7: color_array, shape_array, size_array = [0,0,1],  4, 1
    else                 : color_array, shape_array, size_array = [0,0,1],  4, 2
    return(color_array, shape_array, size_array)

#version2, condensed search::
stimparam_dict = { '1' : [[1,0,0], 40, 1], '2' : [[1,0,0], 40, 2],
                   '3' : [[1,0,0],  4, 1], '4' : [[1,0,0],  4, 2],
                   '5' : [[0,0,1], 40, 1], '6' : [[0,0,1], 40, 2],
                   '7' : [[0,0,1],  4, 1], '8' : [[0,0,1],  4, 2] }
				   
def StimByIdx2(stimulus_id):
    formatted_key = str(stimulus_id)
    color_array   = stimparam_dict[formatted_key][0]
    shape_array   = stimparam_dict[formatted_key][1]
    size_array    = stimparam_dict[formatted_key][2]
    return(color_array, shape_array, size_array)

# Answer Keys
# the last thing we need is a way to decode the <corrresp> column. This holds 
# values from 1-4, which correspond to the 'j'->';' keys. 
def AnswerKeyDecoder2(cor_idx):
    cor_idx =  int(cor_idx)                
    if   cor_idx == 1: correct_resp = 'j'
    elif cor_idx == 2: correct_resp = 'k'
    elif cor_idx == 3: correct_resp = 'l'
    else             : correct_resp = ';'    
    return(correct_resp)
corresp_dict = { '1' : 'j', '2' : 'k', '3' : 'l', '4' : ';' }
# turn the INT into a string, for key matching
def AnswerKeyDecoder(cor_idx):
    str_cor_idx  = str(cor_idx)  
    correct_resp = corresp_dict[str_cor_idx]
    return(correct_resp)
# Sequence ID translator
# at the start of a new block, (training 2 and on) retrieve info for the current
# sequence, and use it to toggle one of the 4 text prompts. from matlab, the
# sequences are (1122, 1221, 2211, 2112) , where 1 = COLOR, and 2 = SHAPE

Seq_Text_dict = { '1' : 'COLOR, COLOR, SHAPE, SHAPE',  # seqidx = 1: 1122
                  '2' : 'COLOR, SHAPE, SHAPE, COLOR',  # seqidx = 2: 1221
                  '3' : 'SHAPE, SHAPE, COLOR, COLOR',  # seqidx = 3: 2211
                  '4' : 'SHAPE, COLOR, COLOR, SHAPE' } # seqidx = 4: 2112

def SeqPrompt(sequence_id):
    formatted_key = str(sequence_id)
    text_sequence = Seq_Text_dict[formatted_key]
    return(text_sequence)

# internal tracking; used at the end of the block for accuracy
corr_count = 0 

# ---------------------------------------------
S12,S34 = 0,0
if S12 == 1:
    for x in [1,2]:
        tr_num = 0
        print('__________________________________________________________')
        while tr_num < 7:
            tr_num += 1
            section_dict = [ S1, S2 ]
            sx = section_dict[ x - 1 ]
            xx = str(x)
            #print(sx)
    
            taskidx = sx[str('taskidx_'+xx)][tr_num]
            color_prompt_yn, shape_prompt_yn = TaskPromptToggle(taskidx)
            
            stimidx = sx[str('stimidx_'+xx)][tr_num]
        #   T1_color, T1_shape, T1_size = StimByIdx(stimidx)
            T1_color, T1_shape, T1_size = StimByIdx2(stimidx)    
            
            corrresp     = sx[str('corrresp_'+xx)][tr_num]
            correct_resp = AnswerKeyDecoder(corrresp)
            
            seqidx      = sx[str('seqidx_'+xx)][tr_num]
            DisplayText = SeqPrompt(seqidx + 1)
            
            print("\n------------ ",'Sec:'+str(x)+'-Tr:'+str(tr_num)," ------------" )
            print('sequence'+str(seqidx)+': ', DisplayText,'color:',T1_color, ' shape:',T1_shape, ' size :',T1_size )
            print('response key:', correct_resp)
elif S34 == 1:
    for x in [3,4]:
        tr_num = 0
        print('__________________________________')
        while tr_num < 12:
            tr_num += 1
            section_dict = [ S3, S4 ]
            sx = section_dict[ x - 3 ]
            xx = str(x)
    
            taskidx  = sx[str('taskidx_'+xx)][tr_num - 1]
            stimidx  = sx[str('stimidx_'+xx)][tr_num - 1]
            corrresp = sx[str('corrresp_'+xx)][tr_num - 1] 
            seqidx   = sx[str('seqidx_'+xx)][tr_num - 1]
            paramidx = [sx[str('color_'+xx)][tr_num - 1]
                       ,sx[str('shape_'+xx)][tr_num - 1]
                       ,sx[str('size_'+xx)][tr_num - 1]]
            posidx   = sx[str('pos_'+xx)][tr_num - 1]

            color_prompt_yn, shape_prompt_yn = TaskPromptToggle(taskidx)
            T1_color, T1_shape, T1_size      = StimByIdx2(stimidx)            
            correct_resp = AnswerKeyDecoder(corrresp)
            DisplayText  = SeqPrompt(seqidx)
            
            print("\n------------ ",'Sec:'+str(x)+'-Tr:'+str(tr_num)," ------------" )
            print('sequence:', DisplayText, 'color:',T1_color,' shape:',T1_shape,' size:',T1_size, 'response key:', correct_resp )
            print(  'pos_idx:',str(seqidx)+':'+str(posidx),
                  'param_idx:',str(paramidx), 'resp_idx:',str(corrresp) )
else: print ('____')


