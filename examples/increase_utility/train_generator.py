import os
import pathlib
from typing import List, Dict
import argparse
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import trlx
from trlx.data.configs import TRLConfig
from utils import get_scores


def load_model(model: str, load_path: str):
    config = AutoConfig.from_pretrained(model, num_labels=1)
    model = AutoModelForSequenceClassification.from_pretrained(model, config=config)
    model.load_state_dict(torch.load(load_path), strict=False)
    return model

# def get_scores(samples: List[str], tokenizer: AutoTokenizer, 
#                model: AutoModelForSequenceClassification) -> torch.tensor:
#     """
#     Could make this more general by adding tokenizer and model as arguments. 
#     That way we can use it for getting gold model scores and reward model scores. 
#     """
#     scores_list = []
#     batch_size = 2
#     for i in range(0, len(samples), batch_size):
#         sub_samples = samples[i: i + batch_size]
#         # sub_samples = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples]
#         encodings_dict = rw_tokenizer(
#             sub_samples,
#             truncation=True,
#             max_length=config.train.seq_length,
#             padding=True,
#             return_tensors="pt",
#         )
#         input_ids = encodings_dict["input_ids"].to(rw_device)
#         attn_masks = encodings_dict["attention_mask"].to(rw_device)
#         input_ids = input_ids.repeat(2, 1)
#         attn_masks = attn_masks.repeat(2, 1)
#         with torch.no_grad():
#             sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks).logits
#             reshaped = sub_scores[:2].reshape(2)
#         scores_list.append(reshaped)
#     scores = torch.cat(scores_list, dim=0)
#     return scores


def get_prompt_dataset(prompts: List[str], max_length: int) -> List[str]:
    formatted_prompts = []
    for i in tqdm(range(len(prompts))):
        # Should we delete the first tmp?
        tmp = tokenizer.decode(
            tokenizer(
                prompts[i],
                truncation=True,
                max_length=max_length,)["input_ids"],
            skip_special_tokens=True,
        ).strip()
        tmp = tokenizer.decode(
            tokenizer(tmp, truncation=True, max_length=max_length)["input_ids"],
            skip_special_tokens=True,
        ).strip()
        formatted_prompts.append(tmp)
    return formatted_prompts


def reward_fn(samples: List[str], normalize=True, **kwargs) -> torch.tensor:
    """
    Normalizes reward scores of completions by subtracting the reward of the original prompt + completion. 
    (`samples`, `prompts`, `outputs`) and the return is a list of `rewards`
    This requires completions, which we won't have for building a proxy model. 
    Maybe replicate Section 2.2 of https://arxiv.org/pdf/2210.10760.pdf 
    """
    scores = get_scores(samples, rw_device, rw_tokenizer, rw_model)
    if normalize:
        original_samples = [text + post_continuation_dict.get(text, " ") for text in samples]
        original_scores = get_scores(original_samples, rw_device, rw_tokenizer, rw_model)
        norms_scores = scores - original_scores
        return norms_scores
    else:
        return scores

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="roberta-large", help="base reward model")
    parser.add_argument("--dataset_path", type=str, default="utility_dataset.hf",
                        help="path to load HuggingFace dataset of continuation pairs")
    parser.add_argument("--reward_checkpoint_path", type=str, default="reward_model/util_roberta-large.pt",
                        help="path to load reward model weights")
    # parser.add_argument("--init_kl_coef", type=float, default=0.1, help="initial KL coefficient")
    parser.add_argument("--config", type=str, default="", help="config letter to use")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    args = parser.parse_args()

    # Load reward model and reward tokenizer
    rw_tokenizer = AutoTokenizer.from_pretrained(args.model)
    rw_tokenizer.pad_token = rw_tokenizer.eos_token
    rw_model = load_model(args.model, args.reward_checkpoint_path)
    rw_device = torch.device("cuda:{}".format(args.gpu))
    rw_model.to(rw_device)
    print("loaded reward model")
    
    # Load another tokenizer -- where's the generator?
    config_path = pathlib.Path(__file__).parent.joinpath(f"configs/ppo_config_gpt2_utility{args.config}.yml")
    config = TRLConfig.load_yaml(config_path)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    max_length_input = (config.train.seq_length - config.method.gen_kwargs["max_new_tokens"])

    # Train from HuggingFace dataset given in commmand line args
    # Question: Not using the continuations, right? We don't need continuations? 
    dataset = load_from_disk(args.dataset_path)
    train_set = [(sample["prompt"], sample["label"]) for sample in dataset["train"]]
    val_set = [(sample["prompt"], sample["label"]) for sample in dataset["val"]]
    train_posts, train_continuations = zip(*train_set)
    val_posts, val_continuations = zip(*val_set)
    post_continuation_dict = {}
    train_prompts = get_prompt_dataset(train_posts, max_length_input)
    for i in range(len(train_prompts)):
        post_continuation_dict[train_prompts[i]] = train_continuations[i]
    val_prompts = get_prompt_dataset(val_posts, max_length_input)
    for i in range(len(val_prompts)):
        post_continuation_dict[val_prompts[i]] = val_continuations[i]
    print("loaded dataset")

    trainer = trlx.train(
        config.model.model_path,  # gpt2
        reward_fn=reward_fn, # How is post_continuation_dict called inside reward_fn without being passed? 
        prompts=train_prompts,
        eval_prompts=train_prompts[0:100],
        config=config,
    )
    # if not os.path.exists("models"):
    #     os.makedirs("models")
    # trainer.save(f'models/gpt2-model2')
    print("done training")
