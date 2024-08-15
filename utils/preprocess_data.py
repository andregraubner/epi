import os
import tarfile
import gzip
import shutil
import polars as pl
import torch

def preprocess(file_path, data_dir):
    # Turn methylation data into sparse pytorch tensors
    df = pl.read_csv(
        file_path, 
        has_header=False, 
        separator="\t",
        new_columns=["chr", "nuc", "pos", "cont", "dinuc", "meth", "mc", "nc"]
    )
    df = df.filter(pl.col("meth").is_not_nan())

    for chromosome, data in df.group_by("chr"):
        chromosome = chromosome[0]

        pos_strand = df.filter(pl.col("nuc") == "C")
        neg_strand = df.filter(pl.col("nuc") == "G")

        pos_indices = pos_strand["pos"].to_torch().unsqueeze(0)
        neg_indices = neg_strand["pos"].to_torch().unsqueeze(0)

        pos_values = (pos_strand["meth"] > 0.5).to_torch()
        neg_values = (neg_strand["meth"] > 0.5).to_torch()

        pos_sparse = torch.sparse_coo_tensor(
            pos_indices, 
            pos_values, 
            size=(len(data),), 
            dtype=torch.bool
        )

        neg_sparse = torch.sparse_coo_tensor(
            neg_indices, 
            neg_values, 
            size=(len(data),), 
            dtype=torch.bool
        )

        # Save the sparse tensor to disk
        basename = os.path.basename(file_path).split('.')[0]
        #os.makedirs(os.path.join(data_dir, chromosome), exist_ok = True) 
        os.makedirs(os.path.join(data_dir, chromosome, "pos_strand"), exist_ok = True) 
        torch.save(pos_sparse, f'data/{chromosome}/pos_strand/{basename}.pth')
        os.makedirs(os.path.join(data_dir, chromosome, "neg_strand"), exist_ok = True) 
        torch.save(neg_sparse, f'data/{chromosome}/neg_strand/{basename}.pth')
    
def extract_tar_gz(file_path, extract_dir):
    # Extract a tar.gz file
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)

def extract_gz(file_path, extract_dir):
    # Extract a single gzipped file
    output_file = os.path.join(extract_dir, os.path.splitext(os.path.basename(file_path))[0])
    with gzip.open(file_path, 'rb') as gz_file:
        with open(output_file, 'wb') as out_file:
            shutil.copyfileobj(gz_file, out_file)

def recursive_extract(directory):
    # Recursively extract all .tar.gz and .gz files in a directory
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.tar.gz'):
                extract_dir = os.path.join(root, os.path.splitext(file)[0])
                os.makedirs(extract_dir, exist_ok=True)
                extract_tar_gz(file_path, extract_dir)
                print(f"Extracted {file_path} to {extract_dir}")
                recursive_extract(extract_dir)  # Recurse into the newly extracted directory
            if file.endswith('.gz'):
                extract_gz(file_path, root)
                print(f"Extracted {file_path} in {root}, opening...")
                preprocess(file_path, "./data")
                quit()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Recursively extract tar.gz and gz files.")
    parser.add_argument("input_file", help="The path to the tar.gz file to be extracted.")
    parser.add_argument("output_dir", help="The directory where the extraction should take place.")
    
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initial extraction
    extract_tar_gz(args.input_file, args.output_dir)
    
    # Recursively handle nested archives
    recursive_extract(args.output_dir)
    
    print("Extraction complete!")