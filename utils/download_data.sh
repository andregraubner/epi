#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <destination_folder>"
  exit 1
fi

# Get the destination folder from the command-line argument
DESTINATION_FOLDER=$1

# Create the destination folder if it doesn't exist
mkdir -p "$DESTINATION_FOLDER"

# Download methylation data
wget -P "$DESTINATION_FOLDER" "https://download.cncb.ac.cn/scmethmap/singlebed/oocyte_Human.tar.gz"

# Download and unzip reference genome data
wget -P "$DESTINATION_FOLDER" "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
gzip -d "$DESTINATION_FOLDER/hg38.fa.gz"

# Get the absolute path of the current bash script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Run the Python script with the downloaded file and the destination folder
python "$SCRIPT_DIR/preprocess_data.py" "$DESTINATION_FOLDER/oocyte_Human.tar.gz" "$DESTINATION_FOLDER"