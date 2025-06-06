"""
Script that runs GCG with multilingual suffix evaluation
"""

import argparse
import os

import torch
import pandas as pd
import random
from   tqdm import tqdm
from   transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   type=str, 
                        default="/hy-tmp/models/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--model_id",     type=str, 
                        default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--device",       type=str, default="cuda")
    parser.add_argument("--dtype",        type=str, default="float16")
    parser.add_argument("--goal_lang",    type=str, default="it")
    parser.add_argument("--target_lang",  type=str, default="en")
    parser.add_argument("--restart",      action="store_true")
    parser.add_argument("--runpod",       action="store_true")
    parser.add_argument("--hy",           action="store_true")
    parser.add_argument("--test_dir",     type=str,  default="./")
    parser.add_argument("--out_dir",      type=str,  default="./")



    args = parser.parse_args()
    return args


def main():
    args         = parse_args()
    goal_lang    = args.goal_lang
    target_lang  = args.target_lang
    restart      = args.restart
    
    model_path   = args.model_path 
    model_id     = args.model_id
    runpod       = args.runpod
    hy           = args.hy
    test_file_dir= args.test_dir
    out_file_dir = args.out_dir

    # Make sure directory exists
    os.makedirs(model_path, exist_ok=True)

    #Instantiate tokenizer and model
    if runpod or hy : 
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 torch_dtype="auto").to("cuda")
    else : 
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=model_path)
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=model_path,
                                                 torch_dtype="auto").to("cuda")

    ling_list= {
        'it':'Italian','vi':'Vietnamese','zh':'Chinese',
        'th':"Thai", 'ko':'Korean', 'ar':'Arabic',
        'sw':'Swahili','bn':'Bengali','jv':'Javanese', 
        'en': 'English'
    }
    
    #test_file_dir = '/workspace/multi_nanoGCG/result'
    test_file     = test_file_dir+ling_list[goal_lang]+'_'+ling_list[target_lang]+'_suffix.csv'
    print (f"(test_file is: {test_file}")
    #out_file_dir  ='/workspace/multi_nanoGCG/result'
    
   

    out_file      = out_file_dir+ling_list[goal_lang]+'_'+ling_list[target_lang]+'_eval_suffix.csv'
    print (f"(out_file is: {out_file}")

    columns       = ['goal','target','ml_goal','ml_target','en_suffix','ml_suffix','loss', 'response']

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
    en_suffix   = df['en_suffix'].tolist()
    ml_suffix   = df['ml_suffix'].tolist()
  
    from tqdm import tqdm
    import re

    for index,(en_goal, ml_goal, en_optim_str, ml_optim_str) in tqdm(enumerate(zip(en_goals, ml_goals, en_suffix, ml_suffix))) :
        eval_goal      = en_goal      if goal_lang   == 'en' else ml_goal
        eval_optim_str = en_optim_str if target_lang == 'en' else ml_optim_str

        adversarial_prompt = eval_goal+' '+eval_optim_str

        print(f"adversarial_prompt -> {adversarial_prompt}")
        # Tokenize
        inputs = tokenizer(adversarial_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
                temperature=1.0,
            )
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'response_text-> {response_text}\n\n')
        
        
        pattern = rf"{re.escape(adversarial_prompt)}"
        new_text = re.sub(pattern, "", response_text)
        
        #print(f'response_text(no prompt)-> {new_text}\n\n')
        new_row = pd.DataFrame([{
            "goal": en_goal,
            "ml_goal": ml_goal,
            "en_suffix": en_optim_str,
            "ml_suffix": ml_optim_str,
            "response": new_text
        }])
        df_output = pd.concat([df_output, new_row], ignore_index=True)        
        if index % 20 == 0 :
            df_output.to_csv(out_file, index=False)
    df_output.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
    #os.system("/usr/bin/shutdown")

