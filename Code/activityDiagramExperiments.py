#%%
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from urllib.parse import unquote

descriptionDFList = []

file_path = 'file//path//to//SystemSpecifications_And_Requirements.xlsx'
resultsDirectoryBase = 'file//path//to//ResultsDirectory' #freely chosen by user

# file_path = 'file:///C:/Users/lasse/Documents/Diss/Code/OCL Constraints und Regeln.xlsx'
# resultsDirectoryBase= 'C:/Users/lasse/Documents/Diss/Experimente'
resultsDirectoryBase = unquote(resultsDirectoryBase)

systemDescriptionKeys = ['_e0','_e3','_e4']
descriptionLanguageKeys = ['_for','_unf']
ruleKeys = ['_5rules','_10rules','_20rules']
taskKeys = ['_base','_cot','_fewShot']
sizeKeys = ['_small','_medium','_large']
modelKeys = ['_3.5t','_4o','_gem','_clau']
#modelKeys = ['_3.5t','_gem']

modelList = []
llm = ChatOpenAI(openai_api_key="insertKeyHere", model='gpt-3.5-turbo-0125', temperature=0)
modelList.append(llm)
llm = ChatOpenAI(openai_api_key="insertKeyHere", model='gpt-4o-2024-05-13', temperature=0)
modelList.append(llm)
llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro',temperature=0, api_key="insertKeyHere")
modelList.append(llm)
llm = ChatAnthropic(model='claude-3-5-sonnet-20240620',temperature=0, api_key="insertKeyHere")
modelList.append(llm)
# llm = ChatOpenAI(model='o1-mini-2024-09-12',temperature=0, openai_api_key="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")
# modelList.append(llm)

modelDict = dict(zip(modelKeys, modelList))


sheet_name = 'intro'
descriptionDF = pd.read_excel(file_path, sheet_name=sheet_name)
intro = descriptionDF.intro[0]

sheet_name = 'small'
descriptionDF = pd.read_excel(file_path, sheet_name=sheet_name)
descriptionDictSDF = dict(zip(systemDescriptionKeys,descriptionDF.SystemDescription))
descriptionDictSDU = dict(zip(systemDescriptionKeys,descriptionDF.SystemDescriptionUnformal))
descriptionDictLanguage = dict(zip(descriptionLanguageKeys,[descriptionDictSDF,descriptionDictSDU]))
descriptionDictE0 = dict(zip(ruleKeys,descriptionDF.e0))
descriptionDictE3 = dict(zip(ruleKeys,descriptionDF.e3))
descriptionDictE4 = dict(zip(ruleKeys,descriptionDF.e4))
descriptionDictListbyError = [descriptionDictE0,descriptionDictE3,descriptionDictE4]
descriptionDictRL = dict(zip(systemDescriptionKeys,descriptionDictListbyError)) 
dictList = [descriptionDictLanguage,descriptionDictRL]
descriptionDFList.append(dictList)

sheet_name = 'medium'
descriptionDF = pd.read_excel(file_path, sheet_name=sheet_name)
descriptionDictSDF = dict(zip(systemDescriptionKeys,descriptionDF.SystemDescription))
descriptionDictSDU = dict(zip(systemDescriptionKeys,descriptionDF.SystemDescriptionUnformal))
descriptionDictLanguage = dict(zip(descriptionLanguageKeys,[descriptionDictSDF,descriptionDictSDU]))
descriptionDictE0 = dict(zip(ruleKeys,descriptionDF.e0))
descriptionDictE3 = dict(zip(ruleKeys,descriptionDF.e3))
descriptionDictE4 = dict(zip(ruleKeys,descriptionDF.e4))
descriptionDictListbyError = [descriptionDictE0,descriptionDictE3,descriptionDictE4]
descriptionDictRL = dict(zip(systemDescriptionKeys,descriptionDictListbyError)) 
dictList = [descriptionDictLanguage,descriptionDictRL]
descriptionDFList.append(dictList)

sheet_name = 'large'
descriptionDF = pd.read_excel(file_path, sheet_name=sheet_name)
descriptionDictSDF = dict(zip(systemDescriptionKeys,descriptionDF.SystemDescription))
descriptionDictSDU = dict(zip(systemDescriptionKeys,descriptionDF.SystemDescriptionUnformal))
descriptionDictLanguage = dict(zip(descriptionLanguageKeys,[descriptionDictSDF,descriptionDictSDU]))
descriptionDictE0 = dict(zip(ruleKeys,descriptionDF.e0))
descriptionDictE3 = dict(zip(ruleKeys,descriptionDF.e3))
descriptionDictE4 = dict(zip(ruleKeys,descriptionDF.e4))
descriptionDictListbyError = [descriptionDictE0,descriptionDictE3,descriptionDictE4]
descriptionDictRL = dict(zip(systemDescriptionKeys,descriptionDictListbyError)) 
dictList = [descriptionDictLanguage,descriptionDictRL]
descriptionDFList.append(dictList)

descriptionDFDict = dict(zip(sizeKeys,descriptionDFList))

#%%

file_path = 'file//path//to//few_shot_example.txt'
file_path = unquote(file_path)
with open(file_path[8:], 'r', encoding='latin-1') as file:
   # Read the entire content of the file into a string variable
   exampleEvalBase = file.read()

#%%
taskList =[]
taskList.append('Is the practical implementation of the workflow valid?')
taskList.append('''Let’s think step by step: does the practical implementation of the Workflow fulfill all the rules stated initially? 
If no rules are broken, the practical implementation can be considered as valid. Please state for every individual rule if and why it is fulfilled or not. 
Also provide a brief summary as to whether all rules are fulfilled.''')
taskList.append('''Let’s think step by step: does the practical implementation of the Workflow fulfill all the rules stated initially? 
If no rules are broken, the practical implementation can be considered as valid. Please state for every individual rule if and why it is fulfilled or not. 
Also provide a brief summary as to whether all rules are fulfilled.''')
taskDict = dict(zip(taskKeys,taskList))


evaluationPrompt = PromptTemplate.from_template(
   '{example} \n Query: \n '+ intro + '\n The following rules must be adhered to: \n {rules}  \n  {workflow} \n {task} \n Answer:')

output_parser = StrOutputParser()

counter = 1    
for sizeKey in sizeKeys:
   for lanKey in descriptionLanguageKeys:
      for sdKey in systemDescriptionKeys:
         for ruleKey in ruleKeys:
            for modelKey in modelKeys:
               for taskKey in taskKeys:
                  if not taskKey == '_fewShot':
                     exampleEval = ' '
                  else:
                     exampleEval = exampleEvalBase
                  llm = modelDict[modelKey]
                  chain = evaluationPrompt | llm | output_parser

                  testPrompt = evaluationPrompt.invoke({
                     "rules": descriptionDFDict[sizeKey][1][sdKey][ruleKey],
                     "workflow": descriptionDFDict[sizeKey][0][lanKey][sdKey],
                     "task": taskDict[taskKey],
                     "example": exampleEval})
                  
                  # llmEvaluation = llm.invoke('which large language model am I talking to?')
                  
                  llmEvaluation = chain.invoke({
                     "rules": descriptionDFDict[sizeKey][1][sdKey][ruleKey],
                     "workflow": descriptionDFDict[sizeKey][0][lanKey][sdKey],
                     "task": taskDict[taskKey],
                     "example": exampleEval})
                  #%%
                  resultFileName = 'evalation'+sizeKey+sdKey+ruleKey+taskKey+modelKey+lanKey
                  resultsDirectory = os.path.join(resultsDirectoryBase, 'activityDiagrams','test_final','Run3')
                  if not os.path.exists(resultsDirectory):
                        os.makedirs(resultsDirectory)
                        print('created Directory: '+resultsDirectory)
                        
                  resultFilePath = os.path.join(resultsDirectory, resultFileName + '.txt')
                  with open(resultFilePath, 'w', encoding='utf-8') as file:
                     file.write(testPrompt.text+' ;;; '+llmEvaluation)
               
                     #file.write(testPrompt.text)
                     print('File written: ' + resultFilePath)
                  
                  print(resultFileName)
                  print("-----")
                  print("-----")
                  print("-----")
                  print("-----")
                  
               # print(counter)
               # counter = counter+1
               # print('-----')
               # print('-----')
# %%
