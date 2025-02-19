# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json

import torch
from torch.utils.data import Dataset

# koalpaca

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        self.ann = json.load(open(dataset_config.data_path))
        # Use 5% of the dataset for evaluation
        eval_length = int(len(self.ann)/20)
        if partition == "train":
            self.ann = self.ann[eval_length:]
        else:
            self.ann = self.ann[:eval_length]

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }
## added----------------------------------------------
'''
####----------------Chunks------------------------------

###corrected

import copy
import json
import torch
from torch.utils.data import Dataset
import re

# Define the prompt format dictionary for LLAMA chat template
PROMPT_DICT = {
    "prompt_input": (
        "<|begin_of_text|><|start_header_id|>System<|end_header_id|>\n\n"
        "{system}<|eot_id|><|start_header_id|>Human<|end_header_id|>\n\n"
        "{instruction}<|eot_id|><|start_header_id|>Assistant<|end_header_id|>\n\n"
    ),
    "prompt_no_input": (
        "<|begin_of_text|><|start_header_id|>System<|end_header_id|>\n\n"
        "{system}<|eot_id|><|start_header_id|>Human<|end_header_id|>\n\n"
        "{instruction}<|eot_id|><|start_header_id|>Assistant<|end_header_id|>\n\n"
    ),
}

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        # Open and read the JSONL file line by line
        with open(dataset_config.data_path, 'r', encoding='utf-8') as jsonl_file:
            self.ann = [json.loads(line) for line in jsonl_file]
        print(f"Total loaded items: {len(self.ann)}")
            
        # Use 8% of the dataset for evaluation
        eval_length = int(len(self.ann) * 0.08)
        if partition == "train":
            self.ann = self.ann[:-eval_length]
        else:
            self.ann = self.ann[-eval_length:]
            
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.ann)
        
    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        ann = self.ann[index]
        text = ann["text"]
        
        # Split into separate conversations using begin_of_text marker
        conversations = text.split("<|begin_of_text|>")
        # Remove empty first element if it exists
        conversations = [conv for conv in conversations if conv.strip()]
        
        # Process all conversations and concatenate them
        full_prompt = ""
        full_output = ""
        
        for conversation in conversations:
            # Prepend the marker that was removed during splitting
            conversation = "<|begin_of_text|>" + conversation
            
            # Extract parts using regex for more robust parsing
            system_pattern = r"<\|start_header_id\|>System<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>"
            human_pattern = r"<\|start_header_id\|>Human<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>"
            assistant_pattern = r"<\|start_header_id\|>Assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>"
            
            # Extract components
            system_match = re.search(system_pattern, conversation, re.DOTALL)
            human_match = re.search(human_pattern, conversation, re.DOTALL)
            assistant_match = re.search(assistant_pattern, conversation, re.DOTALL)
            
            if system_match and human_match and assistant_match:
                system = system_match.group(1).strip()
                instruction = human_match.group(1).strip()
                output = assistant_match.group(1).strip()
                
                # Format prompt for this conversation
                prompt = PROMPT_DICT["prompt_input"].format(
                    system=system,
                    instruction=instruction
                )
                
                full_prompt += prompt
                full_output += output + "<|eot_id|>"
        
        # Combine all prompts with outputs
        example_text = full_prompt + full_output
        
        # Tokenize prompt and full example
        prompt_tokens = self.tokenizer.encode(full_prompt)
        prompt = torch.tensor(prompt_tokens, dtype=torch.int64)
        
        example_tokens = self.tokenizer.encode(example_text)
        # Add EOS token if not already present
        if example_tokens[-1] != self.tokenizer.eos_token_id:
            example_tokens.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example_tokens, dtype=torch.int64)
        
        # Create labels: IGNORE_INDEX for prompt tokens, actual tokens for response
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = IGNORE_INDEX
        
        # Create attention mask (all tokens are attended to)
        example_mask = torch.ones_like(example, dtype=torch.bool)
        
        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask": example_mask.tolist(),
        }
#######----------------------llama-3 chat template------------------------------------

import copy
import json
import torch
from torch.utils.data import Dataset

# Define the prompt format dictionary for LLAMA chat template
PROMPT_DICT = {
    "prompt_input": (
        "<|begin_of_text|><|start_header_id|>System<|end_header_id|>\n\n"
        "{system}<|eot_id|><|start_header_id|>Human<|end_header_id|>\n\n"
        "{instruction}<|eot_id|><|start_header_id|>Assistant<|end_header_id|>\n\n"
    ),
    "prompt_no_input": (
        "<|begin_of_text|><|start_header_id|>System<|end_header_id|>\n\n"
        "{system}<|eot_id|><|start_header_id|>Human<|end_header_id|>\n\n"
        "{instruction}<|eot_id|><|start_header_id|>Assistant<|end_header_id|>\n\n"
    ),
}

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        # Open and read the JSONL file line by line
        with open(dataset_config.data_path, 'r', encoding='utf-8') as jsonl_file:
            self.ann = [json.loads(line) for line in jsonl_file]

        print(int(len(self.ann)))
            
        # Use 0.5% of the dataset for evaluation
        # eval_length = int(len(self.ann))
        eval_length = int(len(self.ann) * 0.08)
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[-eval_length:]
            
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.ann)
        
    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        ann = self.ann[index]
        text = ann["text"]
        
        # Split the text based on LLAMA chat template markers
        parts = text.split("<|start_header_id|>")
        
        # Extract system message (between system and user)
        system_part = parts[1].split("<|eot_id|>")[0]
        system = system_part.split("<|end_header_id|>\n\n")[1].strip()
        
        # Extract user message (between user and assistant)
        user_part = parts[2].split("<|eot_id|>")[0]
        instruction = user_part.split("<|end_header_id|>\n\n")[1].strip()
        
        # Extract assistant message
        assistant_part = parts[3].split("<|eot_id|>")[0]
        output = assistant_part.split("<|end_header_id|>\n\n")[1].strip()
        
        # Format using PROMPT_DICT
        prompt = PROMPT_DICT["prompt_input"].format(
            system=system,
            instruction=instruction
        )
        
        # Combine prompt with output and add EOT token
        example = prompt + output + "<|eot_id|>"
        
        # Tokenize prompt and full example
        prompt = torch.tensor(
            self.tokenizer.encode(prompt),
            dtype=torch.int64
        )
        
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example,
            dtype=torch.int64
        )
        
        # Create labels: -1 for prompt tokens, actual tokens for response
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        
        # Create attention mask
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        
        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask": example_mask.tolist(),
        }
'''
