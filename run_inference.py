'''
Adapted from https://github.com/kojima-takeshi188/zero_shot_cot
'''

import argparse
from utils import *
import pickle

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    
    fix_seed(args.random_seed)
    
    # print("OPENAI_API_KEY:")
    # print(os.getenv("OPENAI_API_KEY")[0:5] + '**********')
    
    # Initialize decoder class (load model and tokenizer) ...
    decoder = HF_Decoder(args)
    
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()

    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method == "few_shot_cot" or args.method == "auto_cot":
        demo = create_demo_text(args, cot_flag=True)
    else:
        pass

    total = 0
    correct_list = []
    generations = {}
    # create experiment folder if not avaibale
    if not os.path.exists('experiment'):
        os.makedirs('experiment', exist_ok=True)
        

    for i, data in enumerate(dataloader):
        # if i < args.resume_id - 1:
        # if i > 10:
        #     continue
        output_line = {}
        
        print('*************************')
        print("{}st data".format(i+1))
                
        # Prepare question template ...
        x, y = data
        x = "Q: " + x[0] + "\n" + "A:"
        y = y[0].strip()
        
        # print(x, y)
        
        output_line["question"] = x
        output_line["gold_ans"] = y

        if args.method == "zero_shot":
            x = x + " " + args.direct_answer_trigger_for_zeroshot
        elif args.method == "zero_shot_cot":
            x = x + " " + args.cot_trigger
        elif args.method == "few_shot":
            x = demo + x
        elif args.method == "few_shot_cot":
            x = demo + x
        elif args.method == "auto_cot":
            x = demo + x + " " + args.cot_trigger
        else:
            raise ValueError("method is not properly defined ...")
        
        # Answer experiment by generating text ...
        max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
        z, log_likelihoods = decoder.decode(args, x, max_length)

        output_line["rationale"] = z
        output_line["token_log_likelihoods"] = log_likelihoods

        # Answer extraction for zero-shot-cot ...
        if args.method == "zero_shot_cot":
            z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
            max_length = args.max_length_direct
            pred, _ = decoder.decode(args, z2, max_length, extract=True)
            # print(z2 + pred)
        else:
            pred = z
            print(x + pred)

        # Clensing of predicted answer ...
        pred = answer_cleansing(args, pred)
        
        
        output_line["pred_ans"] = pred
        output_line["wrap_que"] = x

        # Choose the most frequent answer from the list ...
        print("pred : {}".format(pred))
        print("GT : " + y)
        print('*************************')
        
        # Checking answer ...
        correct = (np.array([pred]) == np.array([y])).sum().item()
        correct_list.append(correct)
        total += 1 #np.array([y]).size(0)
        
        output_line["correct"] = correct
        generations[i] = output_line
        
        if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
            break
            #raise ValueError("Stop !!")


    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))
    
    # save pickle
    with open(args.output_dir, 'wb') as f:
        pickle.dump(generations, f)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="multiarith", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "sarcasm", "svamp", "singleeq", "coin_flip", "last_letters", "riddlesense", "brainteaser", "macgyver"], help="dataset used for experiment"
    )
    parser.add_argument(
        "--demo_path", type=str, default="demos/multiarith", help="pre-generated demos used for experiment"
    )
    parser.add_argument(
        "--resume_id", type=int, default=0, help="resume from which question id (current line number in the output file), if the experiment fails accidently (e.g., network error)"
    )
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")
    
    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")
    
    parser.add_argument("--model_path", type=str, default="/wudi/gysun/init_weights/", help="model path")
    parser.add_argument(
        "--model", type=str, default="gpt3-xl", choices=["gpt3", "Meta-Llama-3-8B-Instruct", "Qwen2-0.5B"], help="model used for decoding. Note that 'gpt3' are the smallest models."
    )
    
    parser.add_argument(
        "--method", type=str, default="auto_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot"], help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="/wudi/gysun/projs/hallucination-agent/experiment/strategyqa.pkl", help="output directory"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=512, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=1024, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help="sleep between runs to avoid excedding the rate limit of openai api"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.9, help="temperature for GPT-3"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "/wudi/gysun/projs/hallucination-agent/dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "sarcasm":
        args.dataset_path = "/wudi/gysun/projs/hallucination-agent/dataset/Sarcasm/sarcasm.jsonl"
        args.direct_answer_trigger = "\nTherefore, is there any sarcasm in this sentence? Please answer Yes or No."
    elif args.dataset == "riddlesense":
        args.dataset_path = "/wudi/gysun/projs/hallucination-agent/dataset/RiddleSense/rs_dev.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "brainteaser":
        args.dataset_path = "/wudi/gysun/projs/hallucination-agent/dataset/BrainTeaser/"
        args.direct_answer_trigger = "\nTherefore, among A through D, the answer is"
    elif args.dataset == "macgyver":
        args.dataset_path = "/wudi/gysun/projs/hallucination-agent/dataset/MacGyver/problem_solution_pair.xlsx"
        args.direct_answer_trigger = "\nTherefore, is the solution solvable? Please answer Yes or No."
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."
    
    return args

if __name__ == "__main__":
    main()