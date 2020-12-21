#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2020.2.4),
    on December 18, 2020, at 13:19
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# using pandas, import the excel file into a dataframe
import pandas as pd

xls = 'Seq2a_test_noblanks.xls'
trial_data = pd.read_excel( xls, ['Sheet1','Sheet2','Sheet3','Sheet4'], index_row=0 )
# store each sheet separately
S1 = trial_data['Sheet1']
S2 = trial_data['Sheet2']
S3 = trial_data['Sheet3']
S4 = trial_data['Sheet4']

# --------- function calls ---------

# Task Training prompts
# We need to toggle  reminder text for the stimuli in training section 1. Here,
# the task_idx indicates which to use: 1= color, 2=shape. These are converted to
# a 1|0 value for (color_prompt_yn, shape_prompt_yn), which the text objects use
# as opacity values. 1 is visible, 0 invisible. 
def TaskPromptToggle(task_id):
    if str(task_id) == '1': colorYN, shapeYN = 1, 0
    else                  : colorYN, shapeYN = 0, 1
    return( colorYN, shapeYN )
    
# Value Conversions; MATLAB idx to rgb, verticies, size
# The MATLAB script handles our variables as 1's and 2's, but we need to do some
# translating before psychopy can use our stimuli. we'll do this by using the 
# stimulus ids, and converting the unique color/shape/size params to something
# psychopy expects to see. Key for below: stim_idx:(color,shape,size).
#1:(1,1,1) 2:(1,1,2) 3:(1,2,1) 4:(1,2,2) 5:(2,1,1) 6:(2,1,2) 7:(2,2,1) 8:(2,2,2)
stimparam_dict = { '1' : [[1,0,0], 40, 1], '2' : [[1,0,0], 40, 2],
                   '3' : [[1,0,0],  4, 1], '4' : [[1,0,0],  4, 2],
                   '5' : [[0,0,1], 40, 1], '6' : [[0,0,1], 40, 2],
                   '7' : [[0,0,1],  4, 1], '8' : [[0,0,1],  4, 2] }
# version2, condensed search instead of if/elif
def StimByIdx(stimulus_id):
    formatted_key = str(stimulus_id)
    color_array   = stimparam_dict[formatted_key][0]
    shape_array   = stimparam_dict[formatted_key][1]
    size_array    = stimparam_dict[formatted_key][2]
    return(color_array, shape_array, size_array)

# Answer Keys
# the last thing we need is a way to decode the <corrresp> column. This holds 
# values from 1-4, which correspond to the 'j'->';' keys. Psychopy doesn't know
# that though, so:
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




# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2020.2.4'
expName = 'Seq2a_4'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sort_keys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='C:\\Seq2a_PsPy\\Seq2a_4.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.DEBUG)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run before the window creation

# Setup the Window
win = visual.Window(
    size=[1536, 864], fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "INIT"
INITClock = core.Clock()
CurrentSect = 0
tr_num = 0
retrainCheck = 1


# Initialize components for Routine "TaskTrain_Explainer"
TaskTrain_ExplainerClock = core.Clock()
T1_Resp = keyboard.Keyboard()
T1_Script = visual.TextStim(win=win, name='T1_Script',
    text='\nTraining (1/3) : Task Structure\n\nFirst, we will practice the decision & response component of the task. \n\nYou will be presented with an object, and should report the COLOR or SHAPEof the object. \nA prompt at the top of the screen will tell you which decision to make for the current object. \nYou will respond primarily with the (J) and (K) keys; please do not take your hand off the keyboard during the task. \n\nTo Begin:\n•  Please place your right hand on the J, K, L, and semicolon ( ; ) keys, with your index finger on the ( J ) key (as if you are typing).\n•  Each time an object appears, press the key corresponding to its color or shape (based on the prompt at the top of the screen)\n•  A cross appears between images; you can ignore this.\n\nPress the space bar when you are ready to begin\n',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=20, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
T1_retrainText = visual.TextStim(win=win, name='T1_retrainText',
    text='                                   *** RETRAIN  ****\nIf you see this text, you may have had trouble with the preceeding task. Please review the instructions, and take your time with the training sections. ',
    font='Arial',
    pos=(0, 20), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1.0, 
    languageStyle='LTR',
    depth=-3.0);

# Initialize components for Routine "fix_cross"
fix_crossClock = core.Clock()
polygon = visual.ShapeStim(
    win=win, name='polygon', vertices='cross',
    size=(0.5, 0.5),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[1,1,1], lineColorSpace='rgb',
    fillColor=[1,1,1], fillColorSpace='rgb',
    opacity=1, depth=0.0, interpolate=True)

# Initialize components for Routine "Training_1"
Training_1Clock = core.Clock()
T1_resp = keyboard.Keyboard()
T1_stim_sq = visual.Rect(
    win=win, name='T1_stim_sq',
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=1.0, lineColorSpace='rgb',
    fillColor=1.0, fillColorSpace='rgb',
    opacity=1.0, depth=-2.0, interpolate=True)
T1_stim_cr = visual.Polygon(
    win=win, name='T1_stim_cr',
    edges=40, size=[1.0, 1.0],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=1.0, lineColorSpace='rgb',
    fillColor=1.0, fillColorSpace='rgb',
    opacity=1.0, depth=-3.0, interpolate=True)
prompt_C = visual.TextBox2(
     win, text='COLOR', font='Arial',
     pos=(0, 10),     letterHeight=0.05,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1.0,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='prompt_C',
     autoLog=True,
)
prompt_S = visual.TextBox2(
     win, text='SHAPE', font='Arial',
     pos=(0, 10),     letterHeight=0.05,
     size=None, borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=1.0,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=None,
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False,
     editable=False,
     name='prompt_S',
     autoLog=True,
)

# Initialize components for Routine "ITI"
ITIClock = core.Clock()
blankText = visual.TextStim(win=win, name='blankText',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=0, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "accuracy_check"
accuracy_checkClock = core.Clock()

# Initialize components for Routine "SeqTrain_Explainer"
SeqTrain_ExplainerClock = core.Clock()
T2_Response = keyboard.Keyboard()
T2_Explainer = visual.TextStim(win=win, name='T2_Explainer',
    text="Training (2/3) : Sequence Training\n\nIn this section you will be shown a 4 item sequence of decisions; and will have to remember which decision to make about each shape.\nFor example, you're given the sequence: 'COLOR', 'SHAPE', 'SHAPE', 'COLOR'.  When the task begins you would respond with the 'COLOR' of the first, \nfollowed by the SHAPE of the second, then the 'SHAPE' of the third, and so on. This will be the rule you follow until the end of the block. \n\nWhen the block ends, you will be presented with a question about the position of the next item in the sequence.\nFor example: The block ended with 'COLOR', 'SHAPE', 'SHAPE',  then the correct response would be '4' because the block ended on the 3rd item in the sequence.\n",
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
T2_retrainText = visual.TextStim(win=win, name='T2_retrainText',
    text='                                   *** RETRAIN  ****\nIf you see this text, you may have had trouble with the preceeding task. Please review the instructions, and take your time with the training sections. ',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1.0, 
    languageStyle='LTR',
    depth=-3.0);

# Initialize components for Routine "SEQ_presentation"
SEQ_presentationClock = core.Clock()
Seq_Prompt = visual.TextStim(win=win, name='Seq_Prompt',
    text='default text',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1.0, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "fix_cross"
fix_crossClock = core.Clock()
polygon = visual.ShapeStim(
    win=win, name='polygon', vertices='cross',
    size=(0.5, 0.5),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[1,1,1], lineColorSpace='rgb',
    fillColor=[1,1,1], fillColorSpace='rgb',
    opacity=1, depth=0.0, interpolate=True)

# Initialize components for Routine "Training_2"
Training_2Clock = core.Clock()
T2_resp = keyboard.Keyboard()
T2_stim_sq = visual.Rect(
    win=win, name='T2_stim_sq',
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=1.0, lineColorSpace='rgb',
    fillColor=1.0, fillColorSpace='rgb',
    opacity=1.0, depth=-2.0, interpolate=True)
T2_stim_cr = visual.Polygon(
    win=win, name='T2_stim_cr',
    edges=40, size=[1.0, 1.0],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=1.0, lineColorSpace='rgb',
    fillColor=1.0, fillColorSpace='rgb',
    opacity=1.0, depth=-3.0, interpolate=True)

# Initialize components for Routine "ITI"
ITIClock = core.Clock()
blankText = visual.TextStim(win=win, name='blankText',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=0, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "accuracy_check"
accuracy_checkClock = core.Clock()

# Initialize components for Routine "Seq_Practice_Explainer"
Seq_Practice_ExplainerClock = core.Clock()
key_resp = keyboard.Keyboard()
T3_Explain = visual.TextStim(win=win, name='T3_Explain',
    text='Any text\n\nincluding line breaks',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);
retrainText = visual.TextStim(win=win, name='retrainText',
    text='                                   *** RETRAIN  ****\nIf you see this text, you may have had trouble with the preceeding task. Please review the instructions, and take your time with the training sections. ',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1.0, 
    languageStyle='LTR',
    depth=-3.0);

# Initialize components for Routine "SEQ_presentation"
SEQ_presentationClock = core.Clock()
Seq_Prompt = visual.TextStim(win=win, name='Seq_Prompt',
    text='default text',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1.0, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "fix_cross"
fix_crossClock = core.Clock()
polygon = visual.ShapeStim(
    win=win, name='polygon', vertices='cross',
    size=(0.5, 0.5),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[1,1,1], lineColorSpace='rgb',
    fillColor=[1,1,1], fillColorSpace='rgb',
    opacity=1, depth=0.0, interpolate=True)

# Initialize components for Routine "Training_3"
Training_3Clock = core.Clock()
T3_resp = keyboard.Keyboard()
T3_stim_sq = visual.Rect(
    win=win, name='T3_stim_sq',
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=1.0, lineColorSpace='rgb',
    fillColor=1.0, fillColorSpace='rgb',
    opacity=1.0, depth=-2.0, interpolate=True)
T3_stim_cr = visual.Polygon(
    win=win, name='T3_stim_cr',
    edges=40, size=[1.0, 1.0],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=T3_color, lineColorSpace='rgb',
    fillColor=T3_color, fillColorSpace='rgb',
    opacity=1.0, depth=-3.0, interpolate=True)

# Initialize components for Routine "ITI"
ITIClock = core.Clock()
blankText = visual.TextStim(win=win, name='blankText',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=0, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "accuracy_check"
accuracy_checkClock = core.Clock()

# Initialize components for Routine "Task_Start_Explainer"
Task_Start_ExplainerClock = core.Clock()
TaskStart_Explain = visual.TextStim(win=win, name='TaskStart_Explain',
    text='\n\nNow we will start the experiment. The only difference from what you just practiced is that the blocks are longer (about 24 shapes).\n\n• Did you get lost at any point? If you can’t remember where you are in the sequence, just decide a place to start in the sequence and go from there. Try not to get flustered and don’t give up!\n• Every set of 4 blocks you will get a break. Feel free to stretch out. Press the space bar to continue when you are done taking a break.\n• There will be 5 sets of blocks total. \n\n',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);
key_resp_3 = keyboard.Keyboard()

# Initialize components for Routine "SEQ_presentation"
SEQ_presentationClock = core.Clock()
Seq_Prompt = visual.TextStim(win=win, name='Seq_Prompt',
    text='default text',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1.0, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "fix_cross"
fix_crossClock = core.Clock()
polygon = visual.ShapeStim(
    win=win, name='polygon', vertices='cross',
    size=(0.5, 0.5),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[1,1,1], lineColorSpace='rgb',
    fillColor=[1,1,1], fillColorSpace='rgb',
    opacity=1, depth=0.0, interpolate=True)

# Initialize components for Routine "Test_1"
Test_1Clock = core.Clock()
key_resp_test = keyboard.Keyboard()
T4_stim_sq = visual.Rect(
    win=win, name='T4_stim_sq',
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=1.0, lineColorSpace='rgb',
    fillColor=1.0, fillColorSpace='rgb',
    opacity=1.0, depth=-2.0, interpolate=True)
T4_stim_cr = visual.Polygon(
    win=win, name='T4_stim_cr',
    edges=40, size=[1.0, 1.0],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=1.0, lineColorSpace='rgb',
    fillColor=1.0, fillColorSpace='rgb',
    opacity=1.0, depth=-3.0, interpolate=True)

# Initialize components for Routine "ITI"
ITIClock = core.Clock()
blankText = visual.TextStim(win=win, name='blankText',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=0, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "seq_pos_prompt"
seq_pos_promptClock = core.Clock()
seqPOS_answer = keyboard.Keyboard()
Test_POS_prompt = visual.TextStim(win=win, name='Test_POS_prompt',
    text='\n           What is the NEXT item in the sequence?\n\n\n\n\n                              1       2        3       4  \n\n                              J       K         L        ; ',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-2.0);

# Initialize components for Routine "Break_Prompt"
Break_PromptClock = core.Clock()
Break_End = keyboard.Keyboard()
Break_Text = visual.TextStim(win=win, name='Break_Text',
    text='If you need a break, now is a good time to pause!\n\n Press space when you are ready to continue.',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "End_Slide"
End_SlideClock = core.Clock()
text = visual.TextStim(win=win, name='text',
    text='Thank you for participating!\n',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "INIT"-------
continueRoutine = True
# update component parameters for each repeat
# keep track of which components have finished
INITComponents = []
for thisComponent in INITComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
INITClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "INIT"-------
while continueRoutine:
    # get current time
    t = INITClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=INITClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in INITComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "INIT"-------
for thisComponent in INITComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "INIT" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
T1_retrain = data.TrialHandler(nReps=retrainCheck, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='T1_retrain')
thisExp.addLoop(T1_retrain)  # add the loop to the experiment
thisT1_retrain = T1_retrain.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisT1_retrain.rgb)
if thisT1_retrain != None:
    for paramName in thisT1_retrain:
        exec('{} = thisT1_retrain[paramName]'.format(paramName))

for thisT1_retrain in T1_retrain:
    currentLoop = T1_retrain
    # abbreviate parameter names if possible (e.g. rgb = thisT1_retrain.rgb)
    if thisT1_retrain != None:
        for paramName in thisT1_retrain:
            exec('{} = thisT1_retrain[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "TaskTrain_Explainer"-------
    continueRoutine = True
    # update component parameters for each repeat
    tr_num = 0            # trial position, local to the current block
    CurrentSect = 1       # id for this section
    sect_corr_ideal   = 8 # max number of correct responses in this section
    total_corr_ideal  = 8 # max # correct responses at this stage in the experiment
    T1_Resp.keys = []
    T1_Resp.rt = []
    _T1_Resp_allKeys = []
    T1_retrainText.setOpacity(retrainCheck)
    # keep track of which components have finished
    TaskTrain_ExplainerComponents = [T1_Resp, T1_Script, T1_retrainText]
    for thisComponent in TaskTrain_ExplainerComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    TaskTrain_ExplainerClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "TaskTrain_Explainer"-------
    while continueRoutine:
        # get current time
        t = TaskTrain_ExplainerClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=TaskTrain_ExplainerClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *T1_Resp* updates
        waitOnFlip = False
        if T1_Resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            T1_Resp.frameNStart = frameN  # exact frame index
            T1_Resp.tStart = t  # local t and not account for scr refresh
            T1_Resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(T1_Resp, 'tStartRefresh')  # time at next scr refresh
            T1_Resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(T1_Resp.clock.reset)  # t=0 on next screen flip
        if T1_Resp.status == STARTED and not waitOnFlip:
            theseKeys = T1_Resp.getKeys(keyList=['y', 'n', 'j', 'k', 'space'], waitRelease=False)
            _T1_Resp_allKeys.extend(theseKeys)
            if len(_T1_Resp_allKeys):
                T1_Resp.keys = _T1_Resp_allKeys[-1].name  # just the last key pressed
                T1_Resp.rt = _T1_Resp_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *T1_Script* updates
        if T1_Script.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            T1_Script.frameNStart = frameN  # exact frame index
            T1_Script.tStart = t  # local t and not account for scr refresh
            T1_Script.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(T1_Script, 'tStartRefresh')  # time at next scr refresh
            T1_Script.setAutoDraw(True)
        
        # *T1_retrainText* updates
        if T1_retrainText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            T1_retrainText.frameNStart = frameN  # exact frame index
            T1_retrainText.tStart = t  # local t and not account for scr refresh
            T1_retrainText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(T1_retrainText, 'tStartRefresh')  # time at next scr refresh
            T1_retrainText.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in TaskTrain_ExplainerComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "TaskTrain_Explainer"-------
    for thisComponent in TaskTrain_ExplainerComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    T1_retrain.addData('T1_Script.started', T1_Script.tStartRefresh)
    T1_retrain.addData('T1_Script.stopped', T1_Script.tStopRefresh)
    T1_retrain.addData('T1_retrainText.started', T1_retrainText.tStartRefresh)
    T1_retrain.addData('T1_retrainText.stopped', T1_retrainText.tStopRefresh)
    # the Routine "TaskTrain_Explainer" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    T1_Trials = data.TrialHandler(nReps=8, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='T1_Trials')
    thisExp.addLoop(T1_Trials)  # add the loop to the experiment
    thisT1_Trial = T1_Trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisT1_Trial.rgb)
    if thisT1_Trial != None:
        for paramName in thisT1_Trial:
            exec('{} = thisT1_Trial[paramName]'.format(paramName))
    
    for thisT1_Trial in T1_Trials:
        currentLoop = T1_Trials
        # abbreviate parameter names if possible (e.g. rgb = thisT1_Trial.rgb)
        if thisT1_Trial != None:
            for paramName in thisT1_Trial:
                exec('{} = thisT1_Trial[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "fix_cross"-------
        continueRoutine = True
        routineTimer.add(1.000000)
        # update component parameters for each repeat
        # keep track of which components have finished
        fix_crossComponents = [polygon]
        for thisComponent in fix_crossComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        fix_crossClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "fix_cross"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = fix_crossClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=fix_crossClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *polygon* updates
            if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon.frameNStart = frameN  # exact frame index
                polygon.tStart = t  # local t and not account for scr refresh
                polygon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                polygon.setAutoDraw(True)
            if polygon.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon.tStop = t  # not accounting for scr refresh
                    polygon.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(polygon, 'tStopRefresh')  # time at next scr refresh
                    polygon.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fix_crossComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "fix_cross"-------
        for thisComponent in fix_crossComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        T1_Trials.addData('polygon.started', polygon.tStartRefresh)
        T1_Trials.addData('polygon.stopped', polygon.tStopRefresh)
        
        # ------Prepare to start Routine "Training_1"-------
        continueRoutine = True
        # update component parameters for each repeat
        # at the start of a Routine (in this case, a Trial), update stim & prompt params
        # using the converters we defined in the Before Experiment tab. 
        taskidx = S1['taskidx_1'][tr_num]
        stimidx = S1['stimidx_1'][tr_num]
        corrresp = S1['corrresp_1'][tr_num]
        
        color_prompt_yn, shape_prompt_yn = TaskPromptToggle(taskidx)
        T1_color, T1_shape, T1_size      = StimByIdx(stimidx)
        correct_resp                     = AnswerKeyDecoder(corrresp)
        
        if T1_shape == 4: cr_visible, sq_visible = 0,1
        else            : cr_visible, sq_visible = 1,0
        
        T1_resp.keys = []
        T1_resp.rt = []
        _T1_resp_allKeys = []
        T1_stim_sq.setOpacity(sq_visible)
        T1_stim_sq.setSize((0.5 * $T1_size, 0.5 * $T1_size))
        T1_stim_sq.setFillColor(T1_color)
        T1_stim_sq.setLineColor(T1_color)
        T1_stim_cr.setOpacity(cr_visible)
        T1_stim_cr.setSize((0.5 * $T1_size, 0.5 * $T1_size))
        T1_stim_cr.setFillColor('#T1_color')
        T1_stim_cr.setLineColor('#T1_color')
        prompt_C.setOpacity(color_prompt_yn)
        prompt_S.setOpacity(shape_prompt_yn)
        # keep track of which components have finished
        Training_1Components = [T1_resp, T1_stim_sq, T1_stim_cr, prompt_C, prompt_S]
        for thisComponent in Training_1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        Training_1Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "Training_1"-------
        while continueRoutine:
            # get current time
            t = Training_1Clock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=Training_1Clock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *T1_resp* updates
            if T1_resp.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                T1_resp.frameNStart = frameN  # exact frame index
                T1_resp.tStart = t  # local t and not account for scr refresh
                T1_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(T1_resp, 'tStartRefresh')  # time at next scr refresh
                T1_resp.status = STARTED
                # keyboard checking is just starting
                T1_resp.clock.reset()  # now t=0
            if T1_resp.status == STARTED:
                theseKeys = T1_resp.getKeys(keyList=['j', 'k'], waitRelease=False)
                _T1_resp_allKeys.extend(theseKeys)
                if len(_T1_resp_allKeys):
                    T1_resp.keys = [key.name for key in _T1_resp_allKeys]  # storing all keys
                    T1_resp.rt = [key.rt for key in _T1_resp_allKeys]
                    # was this correct?
                    if (T1_resp.keys == str(correct_resp)) or (T1_resp.keys == correct_resp):
                        T1_resp.corr = 1
                    else:
                        T1_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *T1_stim_sq* updates
            if T1_stim_sq.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                T1_stim_sq.frameNStart = frameN  # exact frame index
                T1_stim_sq.tStart = t  # local t and not account for scr refresh
                T1_stim_sq.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(T1_stim_sq, 'tStartRefresh')  # time at next scr refresh
                T1_stim_sq.setAutoDraw(True)
            
            # *T1_stim_cr* updates
            if T1_stim_cr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                T1_stim_cr.frameNStart = frameN  # exact frame index
                T1_stim_cr.tStart = t  # local t and not account for scr refresh
                T1_stim_cr.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(T1_stim_cr, 'tStartRefresh')  # time at next scr refresh
                T1_stim_cr.setAutoDraw(True)
            
            # *prompt_C* updates
            if prompt_C.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prompt_C.frameNStart = frameN  # exact frame index
                prompt_C.tStart = t  # local t and not account for scr refresh
                prompt_C.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prompt_C, 'tStartRefresh')  # time at next scr refresh
                prompt_C.setAutoDraw(True)
            
            # *prompt_S* updates
            if prompt_S.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prompt_S.frameNStart = frameN  # exact frame index
                prompt_S.tStart = t  # local t and not account for scr refresh
                prompt_S.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prompt_S, 'tStartRefresh')  # time at next scr refresh
                prompt_S.setAutoDraw(True)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Training_1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "Training_1"-------
        for thisComponent in Training_1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store the last trial params
        last_color_prompt, last_shape_prompt = color_prompt_yn, shape_prompt_yn
        last_color, last_shape, last_size    = T1_color, T1_shape, T1_size
        
        tr_num += 1
        # check responses
        if T1_resp.keys in ['', [], None]:  # No response was made
            T1_resp.keys = None
            # was no response the correct answer?!
            if str(correct_resp).lower() == 'none':
               T1_resp.corr = 1;  # correct non-response
            else:
               T1_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for T1_Trials (TrialHandler)
        T1_Trials.addData('T1_resp.keys',T1_resp.keys)
        T1_Trials.addData('T1_resp.corr', T1_resp.corr)
        if T1_resp.keys != None:  # we had a response
            T1_Trials.addData('T1_resp.rt', T1_resp.rt)
        T1_Trials.addData('T1_resp.started', T1_resp.tStart)
        T1_Trials.addData('T1_resp.stopped', T1_resp.tStop)
        T1_Trials.addData('T1_stim_sq.started', T1_stim_sq.tStartRefresh)
        T1_Trials.addData('T1_stim_sq.stopped', T1_stim_sq.tStopRefresh)
        T1_Trials.addData('T1_stim_cr.started', T1_stim_cr.tStartRefresh)
        T1_Trials.addData('T1_stim_cr.stopped', T1_stim_cr.tStopRefresh)
        T1_Trials.addData('prompt_C.started', prompt_C.tStartRefresh)
        T1_Trials.addData('prompt_C.stopped', prompt_C.tStopRefresh)
        T1_Trials.addData('prompt_S.started', prompt_S.tStartRefresh)
        T1_Trials.addData('prompt_S.stopped', prompt_S.tStopRefresh)
        # the Routine "Training_1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # ------Prepare to start Routine "ITI"-------
        continueRoutine = True
        routineTimer.add(0.500000)
        # update component parameters for each repeat
        # keep track of which components have finished
        ITIComponents = [blankText]
        for thisComponent in ITIComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        ITIClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "ITI"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = ITIClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=ITIClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *blankText* updates
            if blankText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                blankText.frameNStart = frameN  # exact frame index
                blankText.tStart = t  # local t and not account for scr refresh
                blankText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blankText, 'tStartRefresh')  # time at next scr refresh
                blankText.setAutoDraw(True)
            if blankText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blankText.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    blankText.tStop = t  # not accounting for scr refresh
                    blankText.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(blankText, 'tStopRefresh')  # time at next scr refresh
                    blankText.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ITIComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "ITI"-------
        for thisComponent in ITIComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        T1_Trials.addData('blankText.started', blankText.tStartRefresh)
        T1_Trials.addData('blankText.stopped', blankText.tStopRefresh)
        thisExp.nextEntry()
        
    # completed 8 repeats of 'T1_Trials'
    
    
    # ------Prepare to start Routine "accuracy_check"-------
    continueRoutine = True
    # update component parameters for each repeat
    # check accuracy. if <80%, walk it back to the section they just did, by setting
    # the retrain loop in motion.
    
    # fancy version, of the form # 'Yes' if fruit == 'Apple' else 'No'
    if ( (1-(corr_count + 0.001)/(sect_corr_ideal + 0.001)) >= 0.2):
        CurrentSect -= 1
        retrainCheck = 2
        blockStart = 1
    else :
        retrainCheck = 1
    
    
    
    # keep track of which components have finished
    accuracy_checkComponents = []
    for thisComponent in accuracy_checkComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    accuracy_checkClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "accuracy_check"-------
    while continueRoutine:
        # get current time
        t = accuracy_checkClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=accuracy_checkClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in accuracy_checkComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "accuracy_check"-------
    for thisComponent in accuracy_checkComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    corr_count = 0
    # the Routine "accuracy_check" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed retrainCheck repeats of 'T1_retrain'


# set up handler to look after randomisation of conditions etc
T2_retrain = data.TrialHandler(nReps=retrainCheck, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='T2_retrain')
thisExp.addLoop(T2_retrain)  # add the loop to the experiment
thisT2_retrain = T2_retrain.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisT2_retrain.rgb)
if thisT2_retrain != None:
    for paramName in thisT2_retrain:
        exec('{} = thisT2_retrain[paramName]'.format(paramName))

for thisT2_retrain in T2_retrain:
    currentLoop = T2_retrain
    # abbreviate parameter names if possible (e.g. rgb = thisT2_retrain.rgb)
    if thisT2_retrain != None:
        for paramName in thisT2_retrain:
            exec('{} = thisT2_retrain[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "SeqTrain_Explainer"-------
    continueRoutine = True
    # update component parameters for each repeat
    tr_num = 0
    CurrentSect = 2
    sect_corr_ideal  = 8
    total_corr_ideal  = 16
    T2_Response.keys = []
    T2_Response.rt = []
    _T2_Response_allKeys = []
    T2_retrainText.setOpacity(retrainCheck)
    # keep track of which components have finished
    SeqTrain_ExplainerComponents = [T2_Response, T2_Explainer, T2_retrainText]
    for thisComponent in SeqTrain_ExplainerComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    SeqTrain_ExplainerClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "SeqTrain_Explainer"-------
    while continueRoutine:
        # get current time
        t = SeqTrain_ExplainerClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=SeqTrain_ExplainerClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *T2_Response* updates
        waitOnFlip = False
        if T2_Response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            T2_Response.frameNStart = frameN  # exact frame index
            T2_Response.tStart = t  # local t and not account for scr refresh
            T2_Response.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(T2_Response, 'tStartRefresh')  # time at next scr refresh
            T2_Response.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(T2_Response.clock.reset)  # t=0 on next screen flip
        if T2_Response.status == STARTED and not waitOnFlip:
            theseKeys = T2_Response.getKeys(keyList=['y', 'n', 'j', 'k', 'space'], waitRelease=False)
            _T2_Response_allKeys.extend(theseKeys)
            if len(_T2_Response_allKeys):
                T2_Response.keys = _T2_Response_allKeys[-1].name  # just the last key pressed
                T2_Response.rt = _T2_Response_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *T2_Explainer* updates
        if T2_Explainer.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            T2_Explainer.frameNStart = frameN  # exact frame index
            T2_Explainer.tStart = t  # local t and not account for scr refresh
            T2_Explainer.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(T2_Explainer, 'tStartRefresh')  # time at next scr refresh
            T2_Explainer.setAutoDraw(True)
        
        # *T2_retrainText* updates
        if T2_retrainText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            T2_retrainText.frameNStart = frameN  # exact frame index
            T2_retrainText.tStart = t  # local t and not account for scr refresh
            T2_retrainText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(T2_retrainText, 'tStartRefresh')  # time at next scr refresh
            T2_retrainText.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in SeqTrain_ExplainerComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "SeqTrain_Explainer"-------
    for thisComponent in SeqTrain_ExplainerComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    T2_retrain.addData('T2_Explainer.started', T2_Explainer.tStartRefresh)
    T2_retrain.addData('T2_Explainer.stopped', T2_Explainer.tStopRefresh)
    T2_retrain.addData('T2_retrainText.started', T2_retrainText.tStartRefresh)
    T2_retrain.addData('T2_retrainText.stopped', T2_retrainText.tStopRefresh)
    # the Routine "SeqTrain_Explainer" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    T2_Trials = data.TrialHandler(nReps=8, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='T2_Trials')
    thisExp.addLoop(T2_Trials)  # add the loop to the experiment
    thisT2_Trial = T2_Trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisT2_Trial.rgb)
    if thisT2_Trial != None:
        for paramName in thisT2_Trial:
            exec('{} = thisT2_Trial[paramName]'.format(paramName))
    
    for thisT2_Trial in T2_Trials:
        currentLoop = T2_Trials
        # abbreviate parameter names if possible (e.g. rgb = thisT2_Trial.rgb)
        if thisT2_Trial != None:
            for paramName in thisT2_Trial:
                exec('{} = thisT2_Trial[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "SEQ_presentation"-------
        continueRoutine = True
        routineTimer.add(4.000000)
        # update component parameters for each repeat
        # using seqidx, toggle one of the text blocks containing the COLOR/SHAPE seq.
        if tr_num == 0: blockStart = 1
        else         : blockStart = 0
        
        seqidx = S2['seqidx_2'][tr_num]
        DisplayText = SeqPrompt(seqidx) 
        
        Seq_Prompt.setOpacity(blockStart)
        Seq_Prompt.setText(DisplayText)
        # keep track of which components have finished
        SEQ_presentationComponents = [Seq_Prompt]
        for thisComponent in SEQ_presentationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        SEQ_presentationClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "SEQ_presentation"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = SEQ_presentationClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=SEQ_presentationClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Seq_Prompt* updates
            if Seq_Prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Seq_Prompt.frameNStart = frameN  # exact frame index
                Seq_Prompt.tStart = t  # local t and not account for scr refresh
                Seq_Prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Seq_Prompt, 'tStartRefresh')  # time at next scr refresh
                Seq_Prompt.setAutoDraw(True)
            if Seq_Prompt.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Seq_Prompt.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    Seq_Prompt.tStop = t  # not accounting for scr refresh
                    Seq_Prompt.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(Seq_Prompt, 'tStopRefresh')  # time at next scr refresh
                    Seq_Prompt.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in SEQ_presentationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "SEQ_presentation"-------
        for thisComponent in SEQ_presentationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        blockStart = 0
        T2_Trials.addData('Seq_Prompt.started', Seq_Prompt.tStartRefresh)
        T2_Trials.addData('Seq_Prompt.stopped', Seq_Prompt.tStopRefresh)
        
        # ------Prepare to start Routine "fix_cross"-------
        continueRoutine = True
        routineTimer.add(1.000000)
        # update component parameters for each repeat
        # keep track of which components have finished
        fix_crossComponents = [polygon]
        for thisComponent in fix_crossComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        fix_crossClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "fix_cross"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = fix_crossClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=fix_crossClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *polygon* updates
            if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon.frameNStart = frameN  # exact frame index
                polygon.tStart = t  # local t and not account for scr refresh
                polygon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                polygon.setAutoDraw(True)
            if polygon.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon.tStop = t  # not accounting for scr refresh
                    polygon.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(polygon, 'tStopRefresh')  # time at next scr refresh
                    polygon.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fix_crossComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "fix_cross"-------
        for thisComponent in fix_crossComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        T2_Trials.addData('polygon.started', polygon.tStartRefresh)
        T2_Trials.addData('polygon.stopped', polygon.tStopRefresh)
        
        # ------Prepare to start Routine "Training_2"-------
        continueRoutine = True
        # update component parameters for each repeat
        taskidx  = S2['taskidx_2'][tr_num]
        stimidx  = S2['stimidx_2'][tr_num]
        corrresp = S2['corrresp_2'][tr_num]
        
        T2_color, T2_shape, T2_size = StimByIdx(stimidx)
        correct_resp                = AnswerKeyDecoder(corrresp)
        
        if T1_shape == 4: cr_visible, sq_visible = 0,1
        else            : cr_visible, sq_visible = 1,0
        T2_resp.keys = []
        T2_resp.rt = []
        _T2_resp_allKeys = []
        T2_stim_sq.setOpacity(sq_visible)
        T2_stim_sq.setSize((0.5 * $T2_size , 0.5 * $T2_size))
        T2_stim_sq.setFillColor(T2_color)
        T2_stim_sq.setLineColor(T2_color)
        T2_stim_cr.setOpacity(cr_visible)
        T2_stim_cr.setSize((0.5 * $T2_size , 0.5 * $T2_size))
        T2_stim_cr.setFillColor(T2_color)
        T2_stim_cr.setLineColor(T2_color)
        # keep track of which components have finished
        Training_2Components = [T2_resp, T2_stim_sq, T2_stim_cr]
        for thisComponent in Training_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        Training_2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "Training_2"-------
        while continueRoutine:
            # get current time
            t = Training_2Clock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=Training_2Clock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *T2_resp* updates
            waitOnFlip = False
            if T2_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                T2_resp.frameNStart = frameN  # exact frame index
                T2_resp.tStart = t  # local t and not account for scr refresh
                T2_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(T2_resp, 'tStartRefresh')  # time at next scr refresh
                T2_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(T2_resp.clock.reset)  # t=0 on next screen flip
            if T2_resp.status == STARTED and not waitOnFlip:
                theseKeys = T2_resp.getKeys(keyList=['j', 'k', 'l', ';'], waitRelease=False)
                _T2_resp_allKeys.extend(theseKeys)
                if len(_T2_resp_allKeys):
                    T2_resp.keys = [key.name for key in _T2_resp_allKeys]  # storing all keys
                    T2_resp.rt = [key.rt for key in _T2_resp_allKeys]
                    # was this correct?
                    if (T2_resp.keys == str(correct_resp)) or (T2_resp.keys == correct_resp):
                        T2_resp.corr = 1
                    else:
                        T2_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *T2_stim_sq* updates
            if T2_stim_sq.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                T2_stim_sq.frameNStart = frameN  # exact frame index
                T2_stim_sq.tStart = t  # local t and not account for scr refresh
                T2_stim_sq.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(T2_stim_sq, 'tStartRefresh')  # time at next scr refresh
                T2_stim_sq.setAutoDraw(True)
            
            # *T2_stim_cr* updates
            if T2_stim_cr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                T2_stim_cr.frameNStart = frameN  # exact frame index
                T2_stim_cr.tStart = t  # local t and not account for scr refresh
                T2_stim_cr.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(T2_stim_cr, 'tStartRefresh')  # time at next scr refresh
                T2_stim_cr.setAutoDraw(True)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Training_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "Training_2"-------
        for thisComponent in Training_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        last_color_prompt, last_shape_prompt = color_prompt_yn, shape_prompt_yn
        last_color, last_shape, last_size    = T2_color, T2_shape, T2_size
        
        corr_count
        
        # check responses
        if T2_resp.keys in ['', [], None]:  # No response was made
            T2_resp.keys = None
            # was no response the correct answer?!
            if str(correct_resp).lower() == 'none':
               T2_resp.corr = 1;  # correct non-response
            else:
               T2_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for T2_Trials (TrialHandler)
        T2_Trials.addData('T2_resp.keys',T2_resp.keys)
        T2_Trials.addData('T2_resp.corr', T2_resp.corr)
        if T2_resp.keys != None:  # we had a response
            T2_Trials.addData('T2_resp.rt', T2_resp.rt)
        T2_Trials.addData('T2_resp.started', T2_resp.tStartRefresh)
        T2_Trials.addData('T2_resp.stopped', T2_resp.tStopRefresh)
        T2_Trials.addData('T2_stim_sq.started', T2_stim_sq.tStartRefresh)
        T2_Trials.addData('T2_stim_sq.stopped', T2_stim_sq.tStopRefresh)
        T2_Trials.addData('T2_stim_cr.started', T2_stim_cr.tStartRefresh)
        T2_Trials.addData('T2_stim_cr.stopped', T2_stim_cr.tStopRefresh)
        # the Routine "Training_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # ------Prepare to start Routine "ITI"-------
        continueRoutine = True
        routineTimer.add(0.500000)
        # update component parameters for each repeat
        # keep track of which components have finished
        ITIComponents = [blankText]
        for thisComponent in ITIComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        ITIClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "ITI"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = ITIClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=ITIClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *blankText* updates
            if blankText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                blankText.frameNStart = frameN  # exact frame index
                blankText.tStart = t  # local t and not account for scr refresh
                blankText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blankText, 'tStartRefresh')  # time at next scr refresh
                blankText.setAutoDraw(True)
            if blankText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blankText.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    blankText.tStop = t  # not accounting for scr refresh
                    blankText.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(blankText, 'tStopRefresh')  # time at next scr refresh
                    blankText.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ITIComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "ITI"-------
        for thisComponent in ITIComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        T2_Trials.addData('blankText.started', blankText.tStartRefresh)
        T2_Trials.addData('blankText.stopped', blankText.tStopRefresh)
        thisExp.nextEntry()
        
    # completed 8 repeats of 'T2_Trials'
    
    
    # ------Prepare to start Routine "accuracy_check"-------
    continueRoutine = True
    # update component parameters for each repeat
    # check accuracy. if <80%, walk it back to the section they just did, by setting
    # the retrain loop in motion.
    
    # fancy version, of the form # 'Yes' if fruit == 'Apple' else 'No'
    if ( (1-(corr_count + 0.001)/(sect_corr_ideal + 0.001)) >= 0.2):
        CurrentSect -= 1
        retrainCheck = 2
        blockStart = 1
    else :
        retrainCheck = 1
    
    
    
    # keep track of which components have finished
    accuracy_checkComponents = []
    for thisComponent in accuracy_checkComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    accuracy_checkClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "accuracy_check"-------
    while continueRoutine:
        # get current time
        t = accuracy_checkClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=accuracy_checkClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in accuracy_checkComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "accuracy_check"-------
    for thisComponent in accuracy_checkComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    corr_count = 0
    # the Routine "accuracy_check" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
# completed retrainCheck repeats of 'T2_retrain'


# set up handler to look after randomisation of conditions etc
T3_retrain = data.TrialHandler(nReps=retrainCheck, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='T3_retrain')
thisExp.addLoop(T3_retrain)  # add the loop to the experiment
thisT3_retrain = T3_retrain.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisT3_retrain.rgb)
if thisT3_retrain != None:
    for paramName in thisT3_retrain:
        exec('{} = thisT3_retrain[paramName]'.format(paramName))

for thisT3_retrain in T3_retrain:
    currentLoop = T3_retrain
    # abbreviate parameter names if possible (e.g. rgb = thisT3_retrain.rgb)
    if thisT3_retrain != None:
        for paramName in thisT3_retrain:
            exec('{} = thisT3_retrain[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "Seq_Practice_Explainer"-------
    continueRoutine = True
    # update component parameters for each repeat
    CurrentSect = 3
    tr_num = 0
    sect_corr_ideal  = 12
    total_corr_ideal  = 28
    
    
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    retrainText.setOpacity(retrainCheck)
    # keep track of which components have finished
    Seq_Practice_ExplainerComponents = [key_resp, T3_Explain, retrainText]
    for thisComponent in Seq_Practice_ExplainerComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    Seq_Practice_ExplainerClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "Seq_Practice_Explainer"-------
    while continueRoutine:
        # get current time
        t = Seq_Practice_ExplainerClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=Seq_Practice_ExplainerClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp* updates
        waitOnFlip = False
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['y', 'n', 'j', 'k', 'space'], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *T3_Explain* updates
        if T3_Explain.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            T3_Explain.frameNStart = frameN  # exact frame index
            T3_Explain.tStart = t  # local t and not account for scr refresh
            T3_Explain.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(T3_Explain, 'tStartRefresh')  # time at next scr refresh
            T3_Explain.setAutoDraw(True)
        
        # *retrainText* updates
        if retrainText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            retrainText.frameNStart = frameN  # exact frame index
            retrainText.tStart = t  # local t and not account for scr refresh
            retrainText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(retrainText, 'tStartRefresh')  # time at next scr refresh
            retrainText.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Seq_Practice_ExplainerComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "Seq_Practice_Explainer"-------
    for thisComponent in Seq_Practice_ExplainerComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    
    
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    T3_retrain.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        T3_retrain.addData('key_resp.rt', key_resp.rt)
    T3_retrain.addData('key_resp.started', key_resp.tStartRefresh)
    T3_retrain.addData('key_resp.stopped', key_resp.tStopRefresh)
    T3_retrain.addData('T3_Explain.started', T3_Explain.tStartRefresh)
    T3_retrain.addData('T3_Explain.stopped', T3_Explain.tStopRefresh)
    T3_retrain.addData('retrainText.started', retrainText.tStartRefresh)
    T3_retrain.addData('retrainText.stopped', retrainText.tStopRefresh)
    # the Routine "Seq_Practice_Explainer" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    T3_Blocks = data.TrialHandler(nReps=3, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='T3_Blocks')
    thisExp.addLoop(T3_Blocks)  # add the loop to the experiment
    thisT3_Block = T3_Blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisT3_Block.rgb)
    if thisT3_Block != None:
        for paramName in thisT3_Block:
            exec('{} = thisT3_Block[paramName]'.format(paramName))
    
    for thisT3_Block in T3_Blocks:
        currentLoop = T3_Blocks
        # abbreviate parameter names if possible (e.g. rgb = thisT3_Block.rgb)
        if thisT3_Block != None:
            for paramName in thisT3_Block:
                exec('{} = thisT3_Block[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "SEQ_presentation"-------
        continueRoutine = True
        routineTimer.add(4.000000)
        # update component parameters for each repeat
        # using seqidx, toggle one of the text blocks containing the COLOR/SHAPE seq.
        if tr_num == 0: blockStart = 1
        else         : blockStart = 0
        
        seqidx = S2['seqidx_2'][tr_num]
        DisplayText = SeqPrompt(seqidx) 
        
        Seq_Prompt.setOpacity(blockStart)
        Seq_Prompt.setText(DisplayText)
        # keep track of which components have finished
        SEQ_presentationComponents = [Seq_Prompt]
        for thisComponent in SEQ_presentationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        SEQ_presentationClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "SEQ_presentation"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = SEQ_presentationClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=SEQ_presentationClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Seq_Prompt* updates
            if Seq_Prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Seq_Prompt.frameNStart = frameN  # exact frame index
                Seq_Prompt.tStart = t  # local t and not account for scr refresh
                Seq_Prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Seq_Prompt, 'tStartRefresh')  # time at next scr refresh
                Seq_Prompt.setAutoDraw(True)
            if Seq_Prompt.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Seq_Prompt.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    Seq_Prompt.tStop = t  # not accounting for scr refresh
                    Seq_Prompt.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(Seq_Prompt, 'tStopRefresh')  # time at next scr refresh
                    Seq_Prompt.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in SEQ_presentationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "SEQ_presentation"-------
        for thisComponent in SEQ_presentationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        blockStart = 0
        T3_Blocks.addData('Seq_Prompt.started', Seq_Prompt.tStartRefresh)
        T3_Blocks.addData('Seq_Prompt.stopped', Seq_Prompt.tStopRefresh)
        
        # set up handler to look after randomisation of conditions etc
        T3_Trials = data.TrialHandler(nReps=4, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='T3_Trials')
        thisExp.addLoop(T3_Trials)  # add the loop to the experiment
        thisT3_Trial = T3_Trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisT3_Trial.rgb)
        if thisT3_Trial != None:
            for paramName in thisT3_Trial:
                exec('{} = thisT3_Trial[paramName]'.format(paramName))
        
        for thisT3_Trial in T3_Trials:
            currentLoop = T3_Trials
            # abbreviate parameter names if possible (e.g. rgb = thisT3_Trial.rgb)
            if thisT3_Trial != None:
                for paramName in thisT3_Trial:
                    exec('{} = thisT3_Trial[paramName]'.format(paramName))
            
            # ------Prepare to start Routine "fix_cross"-------
            continueRoutine = True
            routineTimer.add(1.000000)
            # update component parameters for each repeat
            # keep track of which components have finished
            fix_crossComponents = [polygon]
            for thisComponent in fix_crossComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            fix_crossClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "fix_cross"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = fix_crossClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=fix_crossClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *polygon* updates
                if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    polygon.frameNStart = frameN  # exact frame index
                    polygon.tStart = t  # local t and not account for scr refresh
                    polygon.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                    polygon.setAutoDraw(True)
                if polygon.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > polygon.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        polygon.tStop = t  # not accounting for scr refresh
                        polygon.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(polygon, 'tStopRefresh')  # time at next scr refresh
                        polygon.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fix_crossComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "fix_cross"-------
            for thisComponent in fix_crossComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            T3_Trials.addData('polygon.started', polygon.tStartRefresh)
            T3_Trials.addData('polygon.stopped', polygon.tStopRefresh)
            
            # ------Prepare to start Routine "Training_3"-------
            continueRoutine = True
            # update component parameters for each repeat
            taskidx  = S3['taskidx_3'][tr_num]
            stimidx  = S3['stimidx_3'][tr_num]
            corrresp = S3['corrresp_3'][tr_num]
            
            T3_color, T3_shape, T3_size = StimByIdx(stimidx)
            correct_resp                = AnswerKeyDecoder(corrresp)
            
            if T1_shape == 4: cr_visible, sq_visible = 0,1
            else            : cr_visible, sq_visible = 1,0
            T3_resp.keys = []
            T3_resp.rt = []
            _T3_resp_allKeys = []
            T3_stim_sq.setOpacity(sq_visible)
            T3_stim_sq.setSize(((0.5 * $T3_size), (0.5 * $T3_size )))
            T3_stim_sq.setFillColor(T3_color)
            T3_stim_sq.setLineColor(T3_color)
            T3_stim_cr.setOpacity(cr_visible)
            T3_stim_cr.setSize(((0.5 * $T3_size), (0.5 * $T3_size )))
            # keep track of which components have finished
            Training_3Components = [T3_resp, T3_stim_sq, T3_stim_cr]
            for thisComponent in Training_3Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            Training_3Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "Training_3"-------
            while continueRoutine:
                # get current time
                t = Training_3Clock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=Training_3Clock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *T3_resp* updates
                waitOnFlip = False
                if T3_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    T3_resp.frameNStart = frameN  # exact frame index
                    T3_resp.tStart = t  # local t and not account for scr refresh
                    T3_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(T3_resp, 'tStartRefresh')  # time at next scr refresh
                    T3_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(T3_resp.clock.reset)  # t=0 on next screen flip
                if T3_resp.status == STARTED and not waitOnFlip:
                    theseKeys = T3_resp.getKeys(keyList=['j', 'k', 'l', ';'], waitRelease=False)
                    _T3_resp_allKeys.extend(theseKeys)
                    if len(_T3_resp_allKeys):
                        T3_resp.keys = _T3_resp_allKeys[-1].name  # just the last key pressed
                        T3_resp.rt = _T3_resp_allKeys[-1].rt
                        # was this correct?
                        if (T3_resp.keys == str(correct_resp)) or (T3_resp.keys == correct_resp):
                            T3_resp.corr = 1
                        else:
                            T3_resp.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                
                # *T3_stim_sq* updates
                if T3_stim_sq.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    T3_stim_sq.frameNStart = frameN  # exact frame index
                    T3_stim_sq.tStart = t  # local t and not account for scr refresh
                    T3_stim_sq.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(T3_stim_sq, 'tStartRefresh')  # time at next scr refresh
                    T3_stim_sq.setAutoDraw(True)
                
                # *T3_stim_cr* updates
                if T3_stim_cr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    T3_stim_cr.frameNStart = frameN  # exact frame index
                    T3_stim_cr.tStart = t  # local t and not account for scr refresh
                    T3_stim_cr.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(T3_stim_cr, 'tStartRefresh')  # time at next scr refresh
                    T3_stim_cr.setAutoDraw(True)
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Training_3Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "Training_3"-------
            for thisComponent in Training_3Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            last_color_prompt, last_shape_prompt = color_prompt_yn, shape_prompt_yn
            last_color, last_shape, last_size    = T3_color, T3_shape, T3_size
            
            if T3_resp.corr == 1: corr_count += 1
            # check responses
            if T3_resp.keys in ['', [], None]:  # No response was made
                T3_resp.keys = None
                # was no response the correct answer?!
                if str(correct_resp).lower() == 'none':
                   T3_resp.corr = 1;  # correct non-response
                else:
                   T3_resp.corr = 0;  # failed to respond (incorrectly)
            # store data for T3_Trials (TrialHandler)
            T3_Trials.addData('T3_resp.keys',T3_resp.keys)
            T3_Trials.addData('T3_resp.corr', T3_resp.corr)
            if T3_resp.keys != None:  # we had a response
                T3_Trials.addData('T3_resp.rt', T3_resp.rt)
            T3_Trials.addData('T3_resp.started', T3_resp.tStartRefresh)
            T3_Trials.addData('T3_resp.stopped', T3_resp.tStopRefresh)
            T3_Trials.addData('T3_stim_sq.started', T3_stim_sq.tStartRefresh)
            T3_Trials.addData('T3_stim_sq.stopped', T3_stim_sq.tStopRefresh)
            T3_Trials.addData('T3_stim_cr.started', T3_stim_cr.tStartRefresh)
            T3_Trials.addData('T3_stim_cr.stopped', T3_stim_cr.tStopRefresh)
            # the Routine "Training_3" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # ------Prepare to start Routine "ITI"-------
            continueRoutine = True
            routineTimer.add(0.500000)
            # update component parameters for each repeat
            # keep track of which components have finished
            ITIComponents = [blankText]
            for thisComponent in ITIComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            ITIClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "ITI"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = ITIClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=ITIClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *blankText* updates
                if blankText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    blankText.frameNStart = frameN  # exact frame index
                    blankText.tStart = t  # local t and not account for scr refresh
                    blankText.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(blankText, 'tStartRefresh')  # time at next scr refresh
                    blankText.setAutoDraw(True)
                if blankText.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > blankText.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        blankText.tStop = t  # not accounting for scr refresh
                        blankText.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(blankText, 'tStopRefresh')  # time at next scr refresh
                        blankText.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in ITIComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "ITI"-------
            for thisComponent in ITIComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            T3_Trials.addData('blankText.started', blankText.tStartRefresh)
            T3_Trials.addData('blankText.stopped', blankText.tStopRefresh)
            thisExp.nextEntry()
            
        # completed 4 repeats of 'T3_Trials'
        
    # completed 3 repeats of 'T3_Blocks'
    
    
    # ------Prepare to start Routine "accuracy_check"-------
    continueRoutine = True
    # update component parameters for each repeat
    # check accuracy. if <80%, walk it back to the section they just did, by setting
    # the retrain loop in motion.
    
    # fancy version, of the form # 'Yes' if fruit == 'Apple' else 'No'
    if ( (1-(corr_count + 0.001)/(sect_corr_ideal + 0.001)) >= 0.2):
        CurrentSect -= 1
        retrainCheck = 2
        blockStart = 1
    else :
        retrainCheck = 1
    
    
    
    # keep track of which components have finished
    accuracy_checkComponents = []
    for thisComponent in accuracy_checkComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    accuracy_checkClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "accuracy_check"-------
    while continueRoutine:
        # get current time
        t = accuracy_checkClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=accuracy_checkClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in accuracy_checkComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "accuracy_check"-------
    for thisComponent in accuracy_checkComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    corr_count = 0
    # the Routine "accuracy_check" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
# completed retrainCheck repeats of 'T3_retrain'


# ------Prepare to start Routine "Task_Start_Explainer"-------
continueRoutine = True
# update component parameters for each repeat
CurrentSect = 4

key_resp_3.keys = []
key_resp_3.rt = []
_key_resp_3_allKeys = []
# keep track of which components have finished
Task_Start_ExplainerComponents = [TaskStart_Explain, key_resp_3]
for thisComponent in Task_Start_ExplainerComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
Task_Start_ExplainerClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "Task_Start_Explainer"-------
while continueRoutine:
    # get current time
    t = Task_Start_ExplainerClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=Task_Start_ExplainerClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *TaskStart_Explain* updates
    if TaskStart_Explain.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        TaskStart_Explain.frameNStart = frameN  # exact frame index
        TaskStart_Explain.tStart = t  # local t and not account for scr refresh
        TaskStart_Explain.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(TaskStart_Explain, 'tStartRefresh')  # time at next scr refresh
        TaskStart_Explain.setAutoDraw(True)
    
    # *key_resp_3* updates
    waitOnFlip = False
    if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_3.frameNStart = frameN  # exact frame index
        key_resp_3.tStart = t  # local t and not account for scr refresh
        key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
        key_resp_3.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
    if key_resp_3.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_3.getKeys(keyList=['space'], waitRelease=False)
        _key_resp_3_allKeys.extend(theseKeys)
        if len(_key_resp_3_allKeys):
            key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
            key_resp_3.rt = _key_resp_3_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in Task_Start_ExplainerComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Task_Start_Explainer"-------
for thisComponent in Task_Start_ExplainerComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('TaskStart_Explain.started', TaskStart_Explain.tStartRefresh)
thisExp.addData('TaskStart_Explain.stopped', TaskStart_Explain.tStopRefresh)
# check responses
if key_resp_3.keys in ['', [], None]:  # No response was made
    key_resp_3.keys = None
thisExp.addData('key_resp_3.keys',key_resp_3.keys)
if key_resp_3.keys != None:  # we had a response
    thisExp.addData('key_resp_3.rt', key_resp_3.rt)
thisExp.addData('key_resp_3.started', key_resp_3.tStartRefresh)
thisExp.addData('key_resp_3.stopped', key_resp_3.tStopRefresh)
thisExp.nextEntry()
# the Routine "Task_Start_Explainer" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
Test_RunLoop = data.TrialHandler(nReps=5, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='Test_RunLoop')
thisExp.addLoop(Test_RunLoop)  # add the loop to the experiment
thisTest_RunLoop = Test_RunLoop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTest_RunLoop.rgb)
if thisTest_RunLoop != None:
    for paramName in thisTest_RunLoop:
        exec('{} = thisTest_RunLoop[paramName]'.format(paramName))

for thisTest_RunLoop in Test_RunLoop:
    currentLoop = Test_RunLoop
    # abbreviate parameter names if possible (e.g. rgb = thisTest_RunLoop.rgb)
    if thisTest_RunLoop != None:
        for paramName in thisTest_RunLoop:
            exec('{} = thisTest_RunLoop[paramName]'.format(paramName))
    
    # set up handler to look after randomisation of conditions etc
    Test_BlockLoop = data.TrialHandler(nReps=4, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='Test_BlockLoop')
    thisExp.addLoop(Test_BlockLoop)  # add the loop to the experiment
    thisTest_BlockLoop = Test_BlockLoop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTest_BlockLoop.rgb)
    if thisTest_BlockLoop != None:
        for paramName in thisTest_BlockLoop:
            exec('{} = thisTest_BlockLoop[paramName]'.format(paramName))
    
    for thisTest_BlockLoop in Test_BlockLoop:
        currentLoop = Test_BlockLoop
        # abbreviate parameter names if possible (e.g. rgb = thisTest_BlockLoop.rgb)
        if thisTest_BlockLoop != None:
            for paramName in thisTest_BlockLoop:
                exec('{} = thisTest_BlockLoop[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "SEQ_presentation"-------
        continueRoutine = True
        routineTimer.add(4.000000)
        # update component parameters for each repeat
        # using seqidx, toggle one of the text blocks containing the COLOR/SHAPE seq.
        if tr_num == 0: blockStart = 1
        else         : blockStart = 0
        
        seqidx = S2['seqidx_2'][tr_num]
        DisplayText = SeqPrompt(seqidx) 
        
        Seq_Prompt.setOpacity(blockStart)
        Seq_Prompt.setText(DisplayText)
        # keep track of which components have finished
        SEQ_presentationComponents = [Seq_Prompt]
        for thisComponent in SEQ_presentationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        SEQ_presentationClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "SEQ_presentation"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = SEQ_presentationClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=SEQ_presentationClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Seq_Prompt* updates
            if Seq_Prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Seq_Prompt.frameNStart = frameN  # exact frame index
                Seq_Prompt.tStart = t  # local t and not account for scr refresh
                Seq_Prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Seq_Prompt, 'tStartRefresh')  # time at next scr refresh
                Seq_Prompt.setAutoDraw(True)
            if Seq_Prompt.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Seq_Prompt.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    Seq_Prompt.tStop = t  # not accounting for scr refresh
                    Seq_Prompt.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(Seq_Prompt, 'tStopRefresh')  # time at next scr refresh
                    Seq_Prompt.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in SEQ_presentationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "SEQ_presentation"-------
        for thisComponent in SEQ_presentationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        blockStart = 0
        Test_BlockLoop.addData('Seq_Prompt.started', Seq_Prompt.tStartRefresh)
        Test_BlockLoop.addData('Seq_Prompt.stopped', Seq_Prompt.tStopRefresh)
        
        # set up handler to look after randomisation of conditions etc
        Task_TrialsLoop = data.TrialHandler(nReps=S4_blocklength, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='Task_TrialsLoop')
        thisExp.addLoop(Task_TrialsLoop)  # add the loop to the experiment
        thisTask_TrialsLoop = Task_TrialsLoop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTask_TrialsLoop.rgb)
        if thisTask_TrialsLoop != None:
            for paramName in thisTask_TrialsLoop:
                exec('{} = thisTask_TrialsLoop[paramName]'.format(paramName))
        
        for thisTask_TrialsLoop in Task_TrialsLoop:
            currentLoop = Task_TrialsLoop
            # abbreviate parameter names if possible (e.g. rgb = thisTask_TrialsLoop.rgb)
            if thisTask_TrialsLoop != None:
                for paramName in thisTask_TrialsLoop:
                    exec('{} = thisTask_TrialsLoop[paramName]'.format(paramName))
            
            # ------Prepare to start Routine "fix_cross"-------
            continueRoutine = True
            routineTimer.add(1.000000)
            # update component parameters for each repeat
            # keep track of which components have finished
            fix_crossComponents = [polygon]
            for thisComponent in fix_crossComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            fix_crossClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "fix_cross"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = fix_crossClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=fix_crossClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *polygon* updates
                if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    polygon.frameNStart = frameN  # exact frame index
                    polygon.tStart = t  # local t and not account for scr refresh
                    polygon.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                    polygon.setAutoDraw(True)
                if polygon.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > polygon.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        polygon.tStop = t  # not accounting for scr refresh
                        polygon.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(polygon, 'tStopRefresh')  # time at next scr refresh
                        polygon.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fix_crossComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "fix_cross"-------
            for thisComponent in fix_crossComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            Task_TrialsLoop.addData('polygon.started', polygon.tStartRefresh)
            Task_TrialsLoop.addData('polygon.stopped', polygon.tStopRefresh)
            
            # ------Prepare to start Routine "Test_1"-------
            continueRoutine = True
            routineTimer.add(4.000000)
            # update component parameters for each repeat
            taskidx  = S4['taskidx_4'][tr_num]
            stimidx  = S4['stimidx_4'][tr_num]
            corrresp = S4['corrresp_4'][tr_num]
            T4_color, T4_shape, T4_size = StimByIdx(stimidx)
            correct_resp                = AnswerKeyDecoder(corrresp)
            
            if T1_shape == 4: cr_visible, sq_visible = 0,1
            else            : cr_visible, sq_visible = 1,0
            key_resp_test.keys = []
            key_resp_test.rt = []
            _key_resp_test_allKeys = []
            T4_stim_sq.setOpacity(sq_visible)
            T4_stim_sq.setSize((0.5 * $T4_size, 0.5 * $T4_size))
            T4_stim_sq.setFillColor(T4_color)
            T4_stim_sq.setLineColor(T4_color)
            T4_stim_cr.setOpacity(cr_visible)
            T4_stim_cr.setSize((0.5 * $T4_size, 0.5 * $T4_size))
            T4_stim_cr.setFillColor(T4_color)
            T4_stim_cr.setLineColor(T4_color)
            # keep track of which components have finished
            Test_1Components = [key_resp_test, T4_stim_sq, T4_stim_cr]
            for thisComponent in Test_1Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            Test_1Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "Test_1"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = Test_1Clock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=Test_1Clock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *key_resp_test* updates
                waitOnFlip = False
                if key_resp_test.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_test.frameNStart = frameN  # exact frame index
                    key_resp_test.tStart = t  # local t and not account for scr refresh
                    key_resp_test.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_test, 'tStartRefresh')  # time at next scr refresh
                    key_resp_test.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_test.clock.reset)  # t=0 on next screen flip
                if key_resp_test.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp_test.tStartRefresh + 4-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp_test.tStop = t  # not accounting for scr refresh
                        key_resp_test.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(key_resp_test, 'tStopRefresh')  # time at next scr refresh
                        key_resp_test.status = FINISHED
                if key_resp_test.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_test.getKeys(keyList=['j', 'k', 'l', ';'], waitRelease=False)
                    _key_resp_test_allKeys.extend(theseKeys)
                    if len(_key_resp_test_allKeys):
                        key_resp_test.keys = [key.name for key in _key_resp_test_allKeys]  # storing all keys
                        key_resp_test.rt = [key.rt for key in _key_resp_test_allKeys]
                        # was this correct?
                        if (key_resp_test.keys == str(corrresp)) or (key_resp_test.keys == corrresp):
                            key_resp_test.corr = 1
                        else:
                            key_resp_test.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                
                # *T4_stim_sq* updates
                if T4_stim_sq.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    T4_stim_sq.frameNStart = frameN  # exact frame index
                    T4_stim_sq.tStart = t  # local t and not account for scr refresh
                    T4_stim_sq.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(T4_stim_sq, 'tStartRefresh')  # time at next scr refresh
                    T4_stim_sq.setAutoDraw(True)
                if T4_stim_sq.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > T4_stim_sq.tStartRefresh + 4-frameTolerance:
                        # keep track of stop time/frame for later
                        T4_stim_sq.tStop = t  # not accounting for scr refresh
                        T4_stim_sq.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(T4_stim_sq, 'tStopRefresh')  # time at next scr refresh
                        T4_stim_sq.setAutoDraw(False)
                
                # *T4_stim_cr* updates
                if T4_stim_cr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    T4_stim_cr.frameNStart = frameN  # exact frame index
                    T4_stim_cr.tStart = t  # local t and not account for scr refresh
                    T4_stim_cr.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(T4_stim_cr, 'tStartRefresh')  # time at next scr refresh
                    T4_stim_cr.setAutoDraw(True)
                if T4_stim_cr.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > T4_stim_cr.tStartRefresh + 4-frameTolerance:
                        # keep track of stop time/frame for later
                        T4_stim_cr.tStop = t  # not accounting for scr refresh
                        T4_stim_cr.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(T4_stim_cr, 'tStopRefresh')  # time at next scr refresh
                        T4_stim_cr.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Test_1Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "Test_1"-------
            for thisComponent in Test_1Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            #get some info on the last trial that happened, incase 
            last_color, last_shape, last_size    = T4_color, T4_shape, T4_size
            
            tr_num += 1
            
            blockCheck = 0
            # check responses
            if key_resp_test.keys in ['', [], None]:  # No response was made
                key_resp_test.keys = None
                # was no response the correct answer?!
                if str(corrresp).lower() == 'none':
                   key_resp_test.corr = 1;  # correct non-response
                else:
                   key_resp_test.corr = 0;  # failed to respond (incorrectly)
            # store data for Task_TrialsLoop (TrialHandler)
            Task_TrialsLoop.addData('key_resp_test.keys',key_resp_test.keys)
            Task_TrialsLoop.addData('key_resp_test.corr', key_resp_test.corr)
            if key_resp_test.keys != None:  # we had a response
                Task_TrialsLoop.addData('key_resp_test.rt', key_resp_test.rt)
            Task_TrialsLoop.addData('key_resp_test.started', key_resp_test.tStartRefresh)
            Task_TrialsLoop.addData('key_resp_test.stopped', key_resp_test.tStopRefresh)
            Task_TrialsLoop.addData('T4_stim_sq.started', T4_stim_sq.tStartRefresh)
            Task_TrialsLoop.addData('T4_stim_sq.stopped', T4_stim_sq.tStopRefresh)
            Task_TrialsLoop.addData('T4_stim_cr.started', T4_stim_cr.tStartRefresh)
            Task_TrialsLoop.addData('T4_stim_cr.stopped', T4_stim_cr.tStopRefresh)
            
            # ------Prepare to start Routine "ITI"-------
            continueRoutine = True
            routineTimer.add(0.500000)
            # update component parameters for each repeat
            # keep track of which components have finished
            ITIComponents = [blankText]
            for thisComponent in ITIComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            ITIClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
            frameN = -1
            
            # -------Run Routine "ITI"-------
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = ITIClock.getTime()
                tThisFlip = win.getFutureFlipTime(clock=ITIClock)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *blankText* updates
                if blankText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    blankText.frameNStart = frameN  # exact frame index
                    blankText.tStart = t  # local t and not account for scr refresh
                    blankText.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(blankText, 'tStartRefresh')  # time at next scr refresh
                    blankText.setAutoDraw(True)
                if blankText.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > blankText.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        blankText.tStop = t  # not accounting for scr refresh
                        blankText.frameNStop = frameN  # exact frame index
                        win.timeOnFlip(blankText, 'tStopRefresh')  # time at next scr refresh
                        blankText.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in ITIComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # -------Ending Routine "ITI"-------
            for thisComponent in ITIComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            Task_TrialsLoop.addData('blankText.started', blankText.tStartRefresh)
            Task_TrialsLoop.addData('blankText.stopped', blankText.tStopRefresh)
            thisExp.nextEntry()
            
        # completed S4_blocklength repeats of 'Task_TrialsLoop'
        
        
        # ------Prepare to start Routine "seq_pos_prompt"-------
        continueRoutine = True
        # update component parameters for each repeat
        posAnswer = AnswerKeyDecoder(pos_4)
        seqPOS_answer.keys = []
        seqPOS_answer.rt = []
        _seqPOS_answer_allKeys = []
        # keep track of which components have finished
        seq_pos_promptComponents = [seqPOS_answer, Test_POS_prompt]
        for thisComponent in seq_pos_promptComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        seq_pos_promptClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "seq_pos_prompt"-------
        while continueRoutine:
            # get current time
            t = seq_pos_promptClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=seq_pos_promptClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *seqPOS_answer* updates
            waitOnFlip = False
            if seqPOS_answer.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                seqPOS_answer.frameNStart = frameN  # exact frame index
                seqPOS_answer.tStart = t  # local t and not account for scr refresh
                seqPOS_answer.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(seqPOS_answer, 'tStartRefresh')  # time at next scr refresh
                seqPOS_answer.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(seqPOS_answer.clock.reset)  # t=0 on next screen flip
            if seqPOS_answer.status == STARTED and not waitOnFlip:
                theseKeys = seqPOS_answer.getKeys(keyList=['j', 'k', 'l', ';'], waitRelease=False)
                _seqPOS_answer_allKeys.extend(theseKeys)
                if len(_seqPOS_answer_allKeys):
                    seqPOS_answer.keys = _seqPOS_answer_allKeys[-1].name  # just the last key pressed
                    seqPOS_answer.rt = _seqPOS_answer_allKeys[-1].rt
                    # was this correct?
                    if (seqPOS_answer.keys == str(corrresp_1)) or (seqPOS_answer.keys == corrresp_1):
                        seqPOS_answer.corr = 1
                    else:
                        seqPOS_answer.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *Test_POS_prompt* updates
            if Test_POS_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Test_POS_prompt.frameNStart = frameN  # exact frame index
                Test_POS_prompt.tStart = t  # local t and not account for scr refresh
                Test_POS_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Test_POS_prompt, 'tStartRefresh')  # time at next scr refresh
                Test_POS_prompt.setAutoDraw(True)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in seq_pos_promptComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "seq_pos_prompt"-------
        for thisComponent in seq_pos_promptComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # check responses
        if seqPOS_answer.keys in ['', [], None]:  # No response was made
            seqPOS_answer.keys = None
            # was no response the correct answer?!
            if str(corrresp_1).lower() == 'none':
               seqPOS_answer.corr = 1;  # correct non-response
            else:
               seqPOS_answer.corr = 0;  # failed to respond (incorrectly)
        # store data for Test_BlockLoop (TrialHandler)
        Test_BlockLoop.addData('seqPOS_answer.keys',seqPOS_answer.keys)
        Test_BlockLoop.addData('seqPOS_answer.corr', seqPOS_answer.corr)
        if seqPOS_answer.keys != None:  # we had a response
            Test_BlockLoop.addData('seqPOS_answer.rt', seqPOS_answer.rt)
        Test_BlockLoop.addData('seqPOS_answer.started', seqPOS_answer.tStartRefresh)
        Test_BlockLoop.addData('seqPOS_answer.stopped', seqPOS_answer.tStopRefresh)
        Test_BlockLoop.addData('Test_POS_prompt.started', Test_POS_prompt.tStartRefresh)
        Test_BlockLoop.addData('Test_POS_prompt.stopped', Test_POS_prompt.tStopRefresh)
        # the Routine "seq_pos_prompt" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 4 repeats of 'Test_BlockLoop'
    
    
    # ------Prepare to start Routine "Break_Prompt"-------
    continueRoutine = True
    # update component parameters for each repeat
    Break_End.keys = []
    Break_End.rt = []
    _Break_End_allKeys = []
    # keep track of which components have finished
    Break_PromptComponents = [Break_End, Break_Text]
    for thisComponent in Break_PromptComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    Break_PromptClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "Break_Prompt"-------
    while continueRoutine:
        # get current time
        t = Break_PromptClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=Break_PromptClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Break_End* updates
        waitOnFlip = False
        if Break_End.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Break_End.frameNStart = frameN  # exact frame index
            Break_End.tStart = t  # local t and not account for scr refresh
            Break_End.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Break_End, 'tStartRefresh')  # time at next scr refresh
            Break_End.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Break_End.clock.reset)  # t=0 on next screen flip
        if Break_End.status == STARTED and not waitOnFlip:
            theseKeys = Break_End.getKeys(keyList=['space'], waitRelease=False)
            _Break_End_allKeys.extend(theseKeys)
            if len(_Break_End_allKeys):
                Break_End.keys = _Break_End_allKeys[-1].name  # just the last key pressed
                Break_End.rt = _Break_End_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *Break_Text* updates
        if Break_Text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Break_Text.frameNStart = frameN  # exact frame index
            Break_Text.tStart = t  # local t and not account for scr refresh
            Break_Text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Break_Text, 'tStartRefresh')  # time at next scr refresh
            Break_Text.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Break_PromptComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "Break_Prompt"-------
    for thisComponent in Break_PromptComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if Break_End.keys in ['', [], None]:  # No response was made
        Break_End.keys = None
    Test_RunLoop.addData('Break_End.keys',Break_End.keys)
    if Break_End.keys != None:  # we had a response
        Test_RunLoop.addData('Break_End.rt', Break_End.rt)
    Test_RunLoop.addData('Break_End.started', Break_End.tStartRefresh)
    Test_RunLoop.addData('Break_End.stopped', Break_End.tStopRefresh)
    Test_RunLoop.addData('Break_Text.started', Break_Text.tStartRefresh)
    Test_RunLoop.addData('Break_Text.stopped', Break_Text.tStopRefresh)
    # the Routine "Break_Prompt" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
# completed 5 repeats of 'Test_RunLoop'


# ------Prepare to start Routine "End_Slide"-------
continueRoutine = True
routineTimer.add(1.000000)
# update component parameters for each repeat
# keep track of which components have finished
End_SlideComponents = [text]
for thisComponent in End_SlideComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
End_SlideClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "End_Slide"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = End_SlideClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=End_SlideClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text* updates
    if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text.frameNStart = frameN  # exact frame index
        text.tStart = t  # local t and not account for scr refresh
        text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
        text.setAutoDraw(True)
    if text.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > text.tStartRefresh + 1.0-frameTolerance:
            # keep track of stop time/frame for later
            text.tStop = t  # not accounting for scr refresh
            text.frameNStop = frameN  # exact frame index
            win.timeOnFlip(text, 'tStopRefresh')  # time at next scr refresh
            text.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in End_SlideComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "End_Slide"-------
for thisComponent in End_SlideComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text.started', text.tStartRefresh)
thisExp.addData('text.stopped', text.tStopRefresh)

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
