from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import polars as pl
import numpy as np
from Bio import SeqIO
import glob

def mlm_getitem(seq, mlm_probability=0.15):
    """Helper method for creating MLM input / target.

    Adapted from:
    https://github.com/huggingface/transformers/blob/14666775a296a76c88e1aa686a9547f393d322e2/src/transformers/data/data_collator.py#L751
    by
    https://github.com/kuleshov-group/caduceus.git
    """
    data = seq.clone()  # remove eos, if applicable
    target = data.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(target.shape, mlm_probability)
    # TODO: Do we need to avoid "masking" special tokens as is done here?
    #  https://github.com/huggingface/transformers/blob/14666775a296a76c88e1aa686a9547f393d322e2/src/transformers/data/data_collator.py#L760-L766
    masked_indices = torch.bernoulli(probability_matrix).bool()
    target[~masked_indices] = 0  # We only compute loss on masked tokens, turn this into pad token

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(target.shape, 0.8)).bool() & masked_indices
    data[indices_replaced] = 6

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(target.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(5, size=target.shape, dtype=torch.long) + 1 # select one of the elegible choices

    data[indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return data, target

def chunk_array(arr, chunk_size):
    return [chunk for chunk in np.array_split(arr, range(chunk_size, len(arr), chunk_size)) if (len(chunk) == chunk_size and np.any(chunk))]

class MethylomeDataset(Dataset):

    def __init__(self, seq_len: int, reference_path: str, methylation_path: str):

        self.tokenizer = {base: i for i, base in enumerate(['N', 'A', 'C', 'G', 'T', 'M', '?'])}

        # Preload and tokenize reference genome
        self.seq_len = seq_len
        hg19 = SeqIO.parse(reference_path, "fasta")
        hg19 = SeqIO.to_dict(hg19)

        chromosomes = list(hg19.keys())

        self.base_seqs = {}
        for chromosome in tqdm(chromosomes):
            base_seq = str(hg19[chromosome].seq).upper()
            base_seq = np.array([self.tokenizer[base] for base in base_seq], dtype=np.uint8)
            self.base_seqs[chromosome] = base_seq


        # Load and correlate methylomes
        self.seqs = []
        for fname in tqdm(glob.glob(methylation_path)):

            # Read .bed files, filter out all methylated cytosines on negative strand for simplicity
            df = pl.read_csv(fname, separator="\t", has_header=False).filter(pl.col("column_6") == "+") # TODO: SORT OUT NEGATIVE STRAND


            for chromosome, data in df.group_by("column_1"):

                chromosome = chromosome[0]
                if chromosome not in chromosomes:
                    continue

                indices = data.select("column_2").to_numpy()[:,0]
                values = (data.select("column_5").to_numpy()[:,0] > 500) # We call the nucleotide methylated if the beta value is > 0.5
                indices = indices[values]

                seq = np.copy(self.base_seqs[chromosome]).astype(np.uint8)
                seq[indices] = 5 # Methylated token is index 5

                seq = chunk_array(seq, chunk_size=seq_len)
                self.seqs.extend(seq)

        print(f"Dataset has {len(self.seqs):_} sequences.")

    def __getitem__(self, idx):

        seq = self.seqs[idx]
        seq = torch.tensor(seq).long()

        data, target = mlm_getitem(seq, mlm_probability=0.15)

        return data, target

    def __len__(self):
        return len(self.seqs)

if __name__ == "__main__":
    dataset = MethylomeDataset(1024)
    print(dataset[0])
