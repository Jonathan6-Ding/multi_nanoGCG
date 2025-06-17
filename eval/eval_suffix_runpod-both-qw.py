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
                        default="/hy-tmp/models/Llama-3.1-8B-Instruct") #models--Qwen--Qwen2.5-7B-Instruct
    parser.add_argument("--model_id",     type=str, 
                        default="meta-llama/Llama-3.1-8B-Instruct") #Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device",       type=str, default="cuda")
    parser.add_argument("--dtype",        type=str, default="float16")
    parser.add_argument("--goal_lang",    type=str, default="it")
    parser.add_argument("--target_lang",  type=str, default="en")
    parser.add_argument("--restart",      action="store_true")
    parser.add_argument("--runpod",       action="store_true")
    parser.add_argument("--hy",           action="store_true")
    parser.add_argument("--test_dir",     type=str,  default="./")
    parser.add_argument("--out_dir",      type=str,  default="./")
    parser.add_argument("--do_qwen",      action="store_true")
    parser.add_argument("--do_llama",     action="store_true")
    parser.add_argument("--do_baseline",  action="store_true")



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
    do_qwen      = args.do_qwen
    do_llama     = args.do_llama
    do_baseline = args.do_baseline
    # Make sure directory exists
    os.makedirs(model_path, exist_ok=True)

    print(f'do_qwen is {do_qwen} and do_llama is {do_llama}')
    
    #Instantiate tokenizer and model
    if do_qwen: 
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        model_path ="/hy-tmp/models/"
    else : 
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        model_path = "/hy-tmp/models/Llama-3.1-8B-Instruct"
   
    print(f'model_id is {model_id}')
    if do_llama : 
        model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto").to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_path,torch_dtype="auto")
    else : 
        model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                 cache_dir= model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto").to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                  cache_dir=model_path,torch_dtype="auto")
    
    ling_list= {
        'it':'Italian','vi':'Vietnamese','zh':'Chinese',
        'th':"Thai", 'ko':'Korean', 'ar':'Arabic',
        'sw':'Swahili','bn':'Bengali','jv':'Javanese', 
        'en': 'English'
    }
    
    #test_file_dir = '/workspace/multi_nanoGCG/result'
    if do_qwen :
        file_prex = 'qwen_'
    elif do_llama : 
        file_prex = 'llama_'
    else :
        file_prex =''
        
    test_file     = test_file_dir+file_prex+ling_list[goal_lang]+'_'+ling_list[target_lang]+'_suffix.csv'

    print (f"(test_file is: {test_file}")
    #out_file_dir  ='/workspace/multi_nanoGCG/result'
    
   

    out_file      = out_file_dir+file_prex+ling_list[goal_lang]+'_'+ling_list[target_lang]+'_both_eval_suffix.csv'
    print (f"(out_file is: {out_file}")

    columns       = ['goal','target','ml_goal','ml_target',
                     'suffix','loss', 'en_response', 'ml_response']

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
    if target_lang == 'en':
        suffixes  = df['en_suffix'].tolist()
    else : 
        suffixes  = df['ml_suffix'].tolist()
    losses        = df['loss'].tolist()
    
    from tqdm import tqdm
    import re

    for index,(en_goal, ml_goal, en_target, ml_target, optim_str, loss) in \
    tqdm(enumerate(zip(en_goals, ml_goals, en_targets, ml_targets, suffixes, losses))) :
        adversarial_prompts = []
        adversarial_prompts.append(en_goal+' '+optim_str)
        if not do_baseline :
            adversarial_prompts.append(ml_goal+' '+optim_str)
 
        new_texts = []
        for adversarial_prompt in adversarial_prompts : 
            print(f"adversarial_prompt -> {adversarial_prompt}")
            # Tokenize
            inputs = tokenizer(adversarial_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                **inputs,
                max_new_tokens=300,
                #do_sample=False,
                #temperature=1.0,
                do_sample=True,
                top_p=0.8,
                top_k=20,
                min_p= 0,
                temperature=0.7,
                repetition_penalty=1.5,
                )
            response_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f'response_text-> {response_text}\n\n')
            pattern = rf"{re.escape(adversarial_prompt)}"
            new_text = re.sub(pattern, "", response_text)
            #print(f'response_text(no prompt)-> {new_text}\n\n')
            new_texts.append(new_text)
        new_row = pd.DataFrame([{
            "goal": en_goal,
            "ml_goal": ml_goal,
            "target" : en_target, 
            "ml_target": ml_target,
            "suffix" : optim_str,
            "loss"   : loss,
            "en_response": new_texts[0],
            "ml_response": new_texts[1] if not do_baseline else new_texts[0]
        }])
        df_output = pd.concat([df_output, new_row], ignore_index=True)        
        if index % 20 == 0 :
            df_output.to_csv(out_file, index=False)
    df_output.to_csv(out_file, index=False)


if __name__ == "__main__":
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    main()
    os.system("shutdown")

