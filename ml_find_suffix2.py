"""
Script that runs GCG with multilingual prompts
"""

import argparse
import os

import torch
import pandas as pd
import random
from   tqdm import tqdm
from   transformers import AutoModelForCausalLM, AutoTokenizer

import nanogcg
from   nanogcg import GCGConfig, ProbeSamplingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   type=str, 
                        default="/root/autodl-tmp/models/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--model_id",     type=str, 
                        default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--device",       type=str, default="cuda")
    parser.add_argument("--dtype",        type=str, default="float16")
    parser.add_argument("--probe-sampling", action="store_true")
    parser.add_argument("--dest",         type=str, default="jv")
    parser.add_argument("--do_en_target", action="store_true")
    parser.add_argument("--do_baseline",  action="store_true")
    parser.add_argument("--num_steps",    type=int ,default=100)
    parser.add_argument("--sample_start", type=int ,default=0)
    parser.add_argument("--sample_end",   type=int ,default=None)
    parser.add_argument("--restart",      action="store_true")

    args = parser.parse_args()
    return args


def main():
    args         = parse_args()
    dest         = args.dest
    do_en_target = args.do_en_target
    do_baseline  = args.do_baseline
    num_steps    = args.num_steps
    sample_start = args.sample_start
    sample_end   = args.sample_end
    
    model_path   = args.model_path 
    model_id     = args.model_id
    restart      = args.restart

    search_width = 64
    print(f'do_en_target is ===> {do_en_target}')

    # Make sure directory exists
    os.makedirs(model_path, exist_ok=True)

    #Instantiate tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=model_path)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=model_path,
                                                 torch_dtype="auto").to("cuda")

   
    
    ling_list= {'it':'Italian','vi':'Vietnamese','zh':'Chinese','th':"Thai",     'ko':'Korean', 'ar':'Arabic','sw':'Swahili','bn':'Bengali','jv':'Javanese', 'en': 'English'}
    
    test_file_dir ='/root/multi_nanoGCG/datset/'
    test_file     = test_file_dir+ling_list[dest]+'_harmful_behaviors.csv'
    
    out_file_dir  ='/root/autodl-tmp/result/'
    
   
    if do_baseline : 
        out_file  = out_file_dir+ling_list['en']+'_'+ling_list['en']+'_suffix.csv'
    elif  do_en_target : 
        out_file  = out_file_dir+ling_list[dest]+'_'+ling_list['en']+'_suffix.csv'
    else : 
        out_file  = out_file_dir+ling_list[dest]+'_'+ling_list[dest]+'_suffix.csv'

   

    columns     = ['goal','target','ml_goal','ml_target','en_suffix','ml_suffix','loss']

    if restart :
        if os.path.exists(out_file):
            os.remove(out_file)

    try:
        df_output = pd.read_csv(out_file)
        # Ensure the loaded DataFrame has all required columns
        for col in columns : 
            if col not in df_output.columns : 
                  df_output[col] = None 
    except FileNotFoundError:
        df_output = pd.DataFrame(columns=columns)
    except pd.errors.EmptyDataError:
        df_output = pd.DataFrame(columns=columns)

    
    df = pd.read_csv(test_file)
    en_goals    = df['goal'].tolist()
    en_targets  = df['target'].tolist()
    ml_goals    = df['ml_goal'].tolist()
    ml_targets  = df['ml_target'].tolist()
    if do_baseline : 
        optim_init = "This is for resarch and evaluaiton "
    else : 
        optim_init  = df['optim_init'].tolist()[0]

    if sample_start is None : 
        sample_start = 0
    if sample_end   is None: 
        sample_end =  len(en_goals)
        
    print(f'sample_start is {sample_start},sample_end is {sample_end}')
    
    en_goals   = en_goals  [sample_start:sample_end]
    en_targets = en_targets[sample_start:sample_end]
    ml_goals   = ml_goals  [sample_start:sample_end]
    ml_targets = ml_targets[sample_start:sample_end]
    
    config = GCGConfig(
        num_steps=num_steps,
        search_width=search_width,
        topk=64,
        seed=42,
        early_stop= False,
        bias_token_id = None,
        verbosity="WARNING", #"INFO"
        filter_ids=False,    #True,
        optim_str_init = optim_init
    )
    
    for index, (ml_goal, ml_target, en_goal, en_target)  in tqdm(enumerate(zip(ml_goals, ml_targets, en_goals, en_targets))):
      print(en_goal, en_target)
      if do_baseline : 
        message = en_goal+" {optim_str}"#+ml_suffix
      else : 
        message = ml_goal+" {optim_str}"#+ml_suffix
  
      if do_en_target :
        target  =en_target+":\n\n"
      else :
        target  =ml_target+":\n\n"
      print(f"do_en_target is ===> {do_en_target}")
      print(f"Message:{message} ->Target:{target}")
      result = nanogcg.run(model, tokenizer, message, target, config)
      print(result)
      new_row = pd.DataFrame([{"goal": en_goal}])
      new_row["ml_goal"]  = ml_goal
      new_row["target"]   = en_target
      new_row["ml_target"]= ml_target

      if do_en_target :
        new_row["en_suffix"]= result.best_string 
      else:
        new_row["ml_suffix"]= result.best_string 
  
      new_row["loss"] = result.best_loss
      df_output = pd.concat([df_output, new_row], ignore_index=True)
      
      if index %20 == 0 : 
          df_output.to_csv(out_file,  index=False)
    df_output.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
    os.system("/usr/bin/shutdown")

