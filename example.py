# -*- coding: utf-8 -*-

import numpy as np
import os
import yaml
from models.basic_methods import DeltaAdditive, MultiplicativeLogarrithm, Division
import json

# Let's say we have following data
seed = 10
np.random.seed(seed)
print("=="*10, 'Step 1. Generate data', "=="*10)
x_curr = np.round(np.random.random((1, 10))*19)
x_base = np.round(np.random.random((1, 10))*19)
column_names = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]

print('x_curr:\n',x_curr,'\nx_base:\n', x_base)

y_curr = np.sum(x_curr)
y_base = np.sum(x_base)
method = 'DeltaAdditive'


# we calculate the contribution value from model DeltaAdditive:
print("\n\n"+"=="*10, 'Step 2. Compute contribution value', "=="*10)
if method == 'DeltaAdditive':
    model = DeltaAdditive()
contribution = model.analysis(x_curr, x_base, y_curr, y_base)
print(contribution, np.sum(contribution))

# get prompt template
print("\n\n"+"=="*10, 'Step 3. Get Prompt Template', "=="*10)
delta_additive_template_file = "/home/yixin/work/msxf/ARIN/prompts/delta_additive.yaml"

with open(delta_additive_template_file) as stream:
    try:
        cfg_file = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
# we get the prompt out of the configure file
prompt = cfg_file['model']['prompt']
# currently, the prompt type is str
print(type(prompt))
print(prompt)

# convert prompt to dictionary
prompt_template = eval(prompt)

# assign the value
# model context
basic_knowledge_file = "/home/yixin/work/msxf/ARIN/documents/basic_knowledge.md"
context_file = "/home/yixin/work/msxf/ARIN/documents/delta_additive.md"
with open(basic_knowledge_file, 'r') as f:
    basic_knowledge = f.read()

with open(context_file, 'r') as f:
    context = f.read()

prompt_template['model_context'] = basic_knowledge + "\n\n" + context
prompt_template['input']['y_curr'] = y_curr 
prompt_template['input']['y_base'] = y_base
prompt_template['input']['x_curr'] = x_curr.tolist() # we have to convert it from numpy array to list
prompt_template['input']['x_base'] = x_base.tolist()
prompt_template['input']['column_names'] = column_names
prompt_template['output'] = contribution.tolist()

hand_to_llm = json.dumps(prompt_template, ensure_ascii=False)



