#Kelly Kung
#read in the training data for emotion
#4/8/17

import numpy as np
import os
import json

def readJsonFiles(path):
    """reads in a Json file and prepares it for the vectorizer"""
    with open(path) as fileName:
        data = json.load(fileName)
    return (data['text'], data['id'], data['in_reply_to_status_id'])

def readFullYVals(path):
    """reads in the truth and skeptcisim json files as a whole"""
    with open(path) as fileName:
        data = json.load(fileName)
    return data 

def readFiles(path):
    """reads the JSON files containing in each folder (and subfolders). 
    The path WD should be set to the WD above the folder semeval2017-task8-dataset.
    Call the function by doing readFiles('semeval2017-task8-dataset').
    
    Returns the text of the json files as well as skepticism levels and truth values"""
    
    JSONdata = dict()
    for directory, folderNames, fileNames in os.walk(path):
        for foldersPath in folderNames:
            if foldersPath == 'traindev':
                for ypath, yfolder, yfiles in os.walk(os.path.join(directory, foldersPath)):
                    for yfile in yfiles:
                        if yfile == 'rumoureval-subtaskA-train.json':
                            skepticism = readFullYVals(os.path.join(ypath, yfile))
                        elif yfile == 'rumoureval-subtaskB-train.json':
                            truthvals = readFullYVals(os.path.join(ypath, yfile))
            elif foldersPath == 'rumoureval-data':
                for dirpath, folders, files in os.walk(os.path.join(directory, foldersPath)):
                    for folder in folders:
                        for dirpath2, folders2, files2 in os.walk(os.path.join(dirpath, folder)):
                            for folderID in folders2:
                                for dirpath3, folders3, files3 in os.walk(os.path.join(dirpath2, folderID)):
                                    for tweetFolder in folders3:
                                        if tweetFolder == 'source-tweet':
                                            for dirpath4, folders4, files4 in os.walk(os.path.join(dirpath3,tweetFolder)):
                                                for fileName in files4:
                                                    jsonText, jsonID, replyID = readJsonFiles(os.path.join(dirpath4,fileName))
                                                    if folderID in JSONdata.keys():
                                                        JSONdata[folderID]['text'] = jsonText
                                                    else:
                                                        JSONdata[folderID] = {'text':jsonText}
                                        elif tweetFolder == 'replies':
                                                for dirpath5, folders5, files5 in os.walk(os.path.join(dirpath3, tweetFolder)):
                                                    for fileName in files5:
                                                        jsonText, jsonID, replyID = readJsonFiles(os.path.join(dirpath5, fileName))
                                                        if folderID in JSONdata.keys():
                                                            JSONdata[folderID][str(jsonID)] = jsonText
                                                        else:
                                                            JSONdata[folderID] = {str(jsonID):jsonText}
    
    return (JSONdata, skepticism, truthvals)
        
def joinJSONdata(jsondata, skepticism, truth):
    dictionary = dict()
    for key in jsondata.keys():
        for subkey in jsondata[key].keys():
            if key not in dictionary.keys() and subkey != 'text':
                if subkey not in skepticism.keys():
                    dictionary[key] = {subkey:{'text':jsondata[key][subkey], 'skepticism':'null'}}
                else:
                    dictionary[key] = {subkey:{'text':jsondata[key][subkey], 'skepticism':skepticism[subkey]}}
            elif key in dictionary.keys() and subkey != 'text':
                if subkey not in skepticism.keys():
                    dictionary[key][subkey] = {'text':jsondata[key][subkey], 'skepticism':'null'}
                else:
                    dictionary[key][subkey] = {'text':jsondata[key][subkey], 'skepticism':skepticism[subkey]}
        if key not in dictionary.keys():
            dictionary[key] = {'text':jsondata[key]['text']}
        else:
            dictionary[key]['text'] = jsondata[key]['text']
        if key not in truth.keys():
            dictionary[key]['truth'] = 'null'
        else:
            dictionary[key]['truth'] = truth[key]
       
    return dictionary

def writeJSON(inFilePath, outFilePath):
    """run this function! takes in the in file path and out file name 
    in order to write a json file containing all the data"""
    jsondata, skept, truth = readFiles(inFilePath)
    joinedData = joinJSONdata(jsondata, skept, truth)
    
    with open(outFilePath, 'wb') as doc:
        json.dump(joinedData, doc)
    
    