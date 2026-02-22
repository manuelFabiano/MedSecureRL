#!/bin/bash
# Download MedMNIST datasets for MedSecure
# Usage: ./scripts/download_data.sh [dataset_name]

set -e

DATASETS=("pathmnist" "chestmnist" "dermamnist" "octmnist" "pneumoniamnist" "bloodmnist")

echo "========================================"
echo "MedSecure - Data Download Script"
echo "========================================"

# Check if specific dataset requested
if [ $# -gt 0 ]; then
    DATASETS=("$@")
fi

# Create data directory
mkdir -p data/raw

# Download datasets using Python
python3 << 'EOF'
import medmnist
from medmnist import INFO
import sys

datasets = sys.argv[1:] if len(sys.argv) > 1 else ["pathmnist", "chestmnist", "dermamnist", "octmnist", "pneumoniamnist", "bloodmnist"]

for name in datasets:
    print(f"\nDownloading {name}...")
    try:
        DataClass = getattr(medmnist, INFO[name]['python_class'])
        # Download train, val, test splits
        for split in ['train', 'val', 'test']:
            _ = DataClass(split=split, download=True, root='./data/raw')
        print(f"  ✓ {name} downloaded successfully")
    except Exception as e:
        print(f"  ✗ Error downloading {name}: {e}")

print("\nDownload complete!")
EOF

echo ""
echo "Data saved to: data/raw/"
echo "========================================"
