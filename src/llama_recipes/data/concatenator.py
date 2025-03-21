# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from tqdm import tqdm
from itertools import chain

from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size

        self.samples = []

        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            }

        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            buffer = {k: v + sample[k] for k,v in buffer.items()}

            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

## added
'''
### WoRKS for all================= chunked data and line-by-line text too :)
class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.samples = []
        
        # Use a simple buffer approach
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        current_length = 0
        
        print("Dataset before Concat:", len(self.dataset))
        print("Type of dataset_train:", type(self.dataset))
        
        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            sample_length = len(sample["input_ids"])
            
            # Check if sample is too large
            if sample_length > self.chunk_size:
                print(f"Warning: Skipping conversation of length {sample_length} (exceeds chunk_size)")
                continue
                
            # If adding this sample would exceed chunk_size, finalize current chunk
            if current_length + sample_length > self.chunk_size:
                # Create new sample from buffer and pad
                padded_sample = self._pad_sample(buffer, current_length)
                self.samples.append(padded_sample)
                
                # Reset buffer
                buffer = {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": [],
                }
                current_length = 0
            
            # Add sample to buffer
            for k, v in sample.items():
                buffer[k].extend(v)
            current_length += sample_length
        
        # Don't forget remaining data in buffer
        if current_length > 0:
            padded_sample = self._pad_sample(buffer, current_length)
            self.samples.append(padded_sample)
            
        print(f"Created {len(self.samples)} chunks from {len(self.dataset)} samples")

    def _pad_sample(self, buffer, current_length):
        """Pad the buffer to reach chunk_size"""
        padding_length = self.chunk_size - current_length
        if padding_length > 0:
            # Create a copy to avoid modifying the buffer
            padded = {k: v.copy() for k, v in buffer.items()}
            
            # Add padding
            padded["input_ids"].extend([0] * padding_length)  # 0 as pad token
            padded["attention_mask"].extend([0] * padding_length)  # 0 for padding positions
            padded["labels"].extend([-100] * padding_length)  # -100 to ignore in loss
            return padded
        else:
            return buffer

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)
'''
## added -------------------------------------------------------------
'''
#------------- padding single text----------------
from tqdm import tqdm
from torch.utils.data import Dataset
import traceback

class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096, tokenizer=None):
        print(f"\nInitializing ConcatDataset:")
        print(f"Dataset length: {len(dataset)}")
        print(f"Chunk size: {chunk_size}")
        print(f"Tokenizer: {tokenizer.__class__.__name__ if tokenizer else None}")
        
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
        self.samples = []
        
        # Try to inspect first sample before processing
        try:
            first_sample = self.dataset[0]
            print("\nFirst sample before processing:")
            print(f"Type: {type(first_sample)}")
            if isinstance(first_sample, dict):
                print("Keys:", first_sample.keys())
                for k, v in first_sample.items():
                    print(f"{k}: type={type(v)}, length={len(v) if hasattr(v, '__len__') else 'N/A'}")
        except Exception as e:
            print(f"Error inspecting first sample: {str(e)}")
            print(traceback.format_exc())
        
        for i, sample in enumerate(tqdm(self.dataset, desc="Processing dataset", dynamic_ncols=True)):
            try:
                # Verify sample structure
                if not isinstance(sample, dict):
                    print(f"Sample {i} is not a dictionary. Type: {type(sample)}")
                    continue
                
                if not all(k in sample for k in ['input_ids', 'attention_mask', 'labels']):
                    missing = [k for k in ['input_ids', 'attention_mask', 'labels'] if k not in sample]
                    print(f"Sample {i} missing required keys: {missing}")
                    continue
                
                # Get the current length of the sequence
                seq_length = len(sample['input_ids'])
                # print(f"\nProcessing sample {i}:")
                # print(f"Sequence length: {seq_length}")
                
                if seq_length < chunk_size:
                    # Calculate padding length
                    pad_length = chunk_size - seq_length
                    # print(f"Adding padding: {pad_length} tokens")
                    
                    # Pad input_ids with tokenizer pad token or 0
                    pad_token = self.tokenizer.pad_token_id if tokenizer else 0
                    padded_input_ids = sample['input_ids'] + [pad_token] * pad_length
                    
                    # Pad attention mask with 0s
                    padded_attention_mask = sample['attention_mask'] + [0] * pad_length
                    
                    # Pad labels with -100
                    padded_labels = sample['labels'] + [-100] * pad_length
                    
                    padded_sample = {
                        'input_ids': padded_input_ids,
                        'attention_mask': padded_attention_mask,
                        'labels': padded_labels
                    }
                    self.samples.append(padded_sample)
                    # print("Successfully added padded sample")
                    
                elif seq_length == chunk_size:
                    # If already at chunk_size, add as is
                    self.samples.append(sample)
                    # print("Added sample without modification")
                
                else:
                    # If longer than chunk_size, warn and truncate
                    print(f"Warning: Sample {i} exceeds chunk_size ({seq_length} > {chunk_size}). Truncating.")
                    truncated_sample = {
                        'input_ids': sample['input_ids'][:chunk_size],
                        'attention_mask': sample['attention_mask'][:chunk_size],
                        'labels': sample['labels'][:chunk_size]
                    }
                    self.samples.append(truncated_sample)
                    print("Successfully added truncated sample")
                
            except Exception as e:
                print(f"\nError processing sample {i}:")
                print(str(e))
                print(traceback.format_exc())
                continue
        
        print(f"\nProcessing complete:")
        print(f"Total samples processed: {len(self.samples)}")
        if len(self.samples) > 0:
            print(f"First sample final length: {len(self.samples[0]['input_ids'])}")

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def __len__(self):
        return len(self.samples)

'''
