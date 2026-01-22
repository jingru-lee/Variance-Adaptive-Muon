"""
Prepare local DCLM shard (already downloaded) + C4/en validation with Llama-2 tokenizer,
tokenize with CPU parallelism, and save HuggingFace Datasets compatible with train_Llama2.py.

Features:
- Step 1: Load Llama-2 tokenizer
- Step 2: Load and process local C4 validation data from ./c4-validation folder
- Step 3: Decompress DCLM .zst files to ./DCLM folder (with caching)
- Step 4: Process DCLM dataset with 8 parallel workers

Example (Bash):
  export HF_TOKEN="your_token"
  python 2.py \
    --dclm_dir ./global-shard_01_of_10 \
    --output_dir ./data \
    --cache_dir autodl-tmp \
    --max_length 4096 \
    --num_proc 8

Then train:
  torchrun --standalone --nproc_per_node 8 train_Llama2.py \
    --model_config configs/llama_1.2b.json \
    ... \
    --dataset_info data/dataset_info_llama2.json
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import time
import gc
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from datetime import datetime


def setup_hf_cache_dirs(cache_dir: str) -> Dict[str, str]:
    """Setup HuggingFace cache directories."""
    cache_dir_abs = os.path.abspath(os.path.join(os.getcwd(), cache_dir))
    hf_home = os.path.join(cache_dir_abs, "huggingface")
    hf_datasets_cache = os.path.join(hf_home, "datasets")
    hf_hub_cache = os.path.join(hf_home, "hub")
    hf_transformers_cache = os.path.join(hf_home, "transformers")

    for p in [hf_home, hf_datasets_cache, hf_hub_cache, hf_transformers_cache]:
        os.makedirs(p, exist_ok=True)

    os.environ["HF_HOME"] = hf_home
    os.environ["HF_DATASETS_CACHE"] = hf_datasets_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache
    os.environ["TRANSFORMERS_CACHE"] = hf_transformers_cache
    os.environ["HF_DATASETS_OFFLINE"] = "0"

    return {
        "HF_HOME": hf_home,
        "HF_DATASETS_CACHE": hf_datasets_cache,
        "HUGGINGFACE_HUB_CACHE": hf_hub_cache,
        "TRANSFORMERS_CACHE": hf_transformers_cache,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Prepare local DCLM shard + C4/en validation with Llama-2 tokenizer (CPU parallel tokenization).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python 2.py --dclm_dir ./global-shard_01_of_10

  # Custom settings with batch processing
  python 2.py \\
    --dclm_dir ./global-shard_01_of_10 \\
    --output_dir ./data \\
    --cache_dir autodl-tmp \\
    --max_length 4096 \\
    --num_proc 8 \\
    --batch_size 10000

  # For debugging with limited samples
  python 2.py \\
    --dclm_dir ./global-shard_01_of_10 \\
    --dclm_max_samples 10000 \\
    --c4_max_samples 1000
        """
    )
    p.add_argument(
        "--dclm_dir",
        type=str,
        required=True,
        help="Local directory containing downloaded DCLM shard files (e.g., ./global-shard_01_of_10).",
    )
    p.add_argument(
        "--c4_validation_dir",
        type=str,
        default="c4-validation",
        help="Local directory containing C4 validation data files (default: ./c4-validation).",
    )
    p.add_argument(
        "--dclm_output_dir",
        type=str,
        default="DCLM",
        help="Directory to store decompressed DCLM files (default: ./DCLM).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save tokenized datasets and dataset_info_llama2.json (default: ./data).",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default="autodl-tmp",
        help="Cache directory (relative to CWD) for HuggingFace hub/datasets/transformers.",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Max sequence length for tokenization (padding to max_length).",
    )
    p.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="CPU parallelism for tokenization via datasets.map(num_proc=...).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Batch size for streaming data loading (number of records per batch).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (used only if enabling shuffle).",
    )
    p.add_argument(
        "--shuffle_train",
        action="store_true",
        help="Whether to shuffle the loaded training dataset (creates indices; can be slow for very large sets).",
    )
    p.add_argument(
        "--c4_max_samples",
        type=int,
        default=None,
        help="Optionally limit number of C4 validation samples for debugging. Default: full validation.",
    )
    p.add_argument(
        "--dclm_max_samples",
        type=int,
        default=None,
        help="Optionally limit number of local DCLM samples for debugging. Default: use all local files.",
    )
    return p.parse_args()


def _find_local_data_files(root: Path) -> Dict[str, List[str]]:
    """Recursively find supported raw files under root."""
    if not root.exists():
        raise FileNotFoundError(f"dclm_dir does not exist: {root}")

    parquet_files = sorted([str(p) for p in root.rglob("*.parquet")])
    jsonl_files = sorted([str(p) for p in root.rglob("*.jsonl")])
    jsonl_files = [f for f in jsonl_files if not f.endswith('.jsonl.zst')]
    json_files = sorted([str(p) for p in root.rglob("*.json")])
    zst_files = sorted([str(p) for p in root.rglob("*.jsonl.zst")])
    
    return {
        "parquet": parquet_files, 
        "jsonl": jsonl_files, 
        "json": json_files,
        "zst": zst_files
    }


def _find_c4_validation_files(root: Path) -> Dict[str, List[str]]:
    """Find C4 validation data files in local directory."""
    if not root.exists():
        raise FileNotFoundError(f"C4 validation directory does not exist: {root}")

    # Support various C4 file formats
    parquet_files = sorted([str(p) for p in root.rglob("*.parquet")])
    jsonl_files = sorted([str(p) for p in root.rglob("*.jsonl")])
    jsonl_files = [f for f in jsonl_files if not f.endswith('.jsonl.zst')]
    json_files = sorted([str(p) for p in root.rglob("*.json")])
    zst_files = sorted([str(p) for p in root.rglob("*.jsonl.zst")])
    gz_files = sorted([str(p) for p in root.rglob("*.json.gz")])
    
    return {
        "parquet": parquet_files, 
        "jsonl": jsonl_files, 
        "json": json_files,
        "zst": zst_files,
        "gz": gz_files
    }


def _check_dclm_decompressed(dclm_output_dir: Path) -> bool:
    """
    Check if DCLM folder already contains decompressed files.
    Returns True if folder exists and contains .jsonl files.
    """
    if not dclm_output_dir.exists():
        return False
    
    # Check for decompressed .jsonl files (including in subdirectories)
    jsonl_files = list(dclm_output_dir.rglob("*.jsonl"))
    
    if len(jsonl_files) > 0:
        print(f"  Found {len(jsonl_files)} decompressed .jsonl files in {dclm_output_dir}")
        return True
    
    return False


def _decompress_zst_to_folder(
    zst_files: List[str], 
    source_root: Path,
    output_dir: Path, 
    max_samples: Optional[int] = None
) -> List[str]:
    """
    Decompress .zst files to output directory, preserving subdirectory structure.
    
    Args:
        zst_files: List of .zst file paths
        source_root: Root directory of the source files (e.g., global-shard_01_of_10)
        output_dir: Directory to save decompressed files (e.g., DCLM)
        max_samples: Optional limit on total samples (not files)
        
    Returns:
        List of decompressed file paths
    """
    try:
        import zstandard as zstd
    except ImportError:
        raise ImportError(
            "zstandard library is required to read .zst files. "
            "Install it with: pip install zstandard"
        )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    decompressed_files = []
    total_samples = 0
    
    # Resolve source_root to absolute path for relative path calculation
    source_root_resolved = source_root.resolve()
    
    print(f"  Decompressing {len(zst_files)} .zst files to {output_dir}...")
    print(f"  Source root: {source_root_resolved}")
    print(f"  Preserving subdirectory structure to avoid filename conflicts...")
    
    for i, zst_file in enumerate(zst_files):
        if max_samples is not None and total_samples >= max_samples:
            print(f"  Reached max_samples limit ({max_samples}), stopping decompression.")
            break
        
        zst_path = Path(zst_file).resolve()
        
        # Calculate relative path from source root
        try:
            relative_path = zst_path.relative_to(source_root_resolved)
        except ValueError:
            # If the file is not under source_root, use just the filename
            relative_path = Path(zst_path.name)
        
        # Get the subdirectory structure (parent of the file relative to source root)
        relative_subdir = relative_path.parent
        
        # Generate output filename (remove .zst extension)
        if zst_path.name.endswith('.jsonl.zst'):
            out_name = zst_path.name[:-4]  # Remove .zst, keep .jsonl
        else:
            out_name = zst_path.stem + ".jsonl"
        
        # Create output subdirectory in DCLM folder, mirroring source structure
        out_subdir = output_dir / relative_subdir
        out_subdir.mkdir(parents=True, exist_ok=True)
        
        out_path = out_subdir / out_name
        
        # Skip if already decompressed
        if out_path.exists():
            print(f"    [{i+1}/{len(zst_files)}] Already exists: {relative_subdir / out_name}")
            decompressed_files.append(str(out_path))
            # Count samples in existing file
            with open(out_path, 'r', encoding='utf-8') as f:
                for _ in f:
                    total_samples += 1
            continue
        
        print(f"    [{i+1}/{len(zst_files)}] Decompressing: {relative_path} -> {relative_subdir / out_name}")
        
        try:
            dctx = zstd.ZstdDecompressor()
            samples_in_file = 0
            
            with open(zst_file, 'rb') as ifh:
                with dctx.stream_reader(ifh) as reader:
                    with open(out_path, 'w', encoding='utf-8') as ofh:
                        # Read in chunks to handle large files
                        text_buffer = ""
                        while True:
                            chunk = reader.read(1024 * 1024)  # 1MB chunks
                            if not chunk:
                                break
                            text_buffer += chunk.decode('utf-8')
                            
                            # Process complete lines
                            lines = text_buffer.split('\n')
                            text_buffer = lines[-1]  # Keep incomplete line
                            
                            for line in lines[:-1]:
                                if line.strip():
                                    ofh.write(line + '\n')
                                    samples_in_file += 1
                                    total_samples += 1
                                    
                                    if max_samples is not None and total_samples >= max_samples:
                                        break
                            
                            if max_samples is not None and total_samples >= max_samples:
                                break
                        
                        # Write remaining buffer
                        if text_buffer.strip() and (max_samples is None or total_samples < max_samples):
                            ofh.write(text_buffer + '\n')
                            samples_in_file += 1
                            total_samples += 1
            
            decompressed_files.append(str(out_path))
            print(f"      -> {samples_in_file} samples")
            
        except Exception as e:
            print(f"    Warning: Failed to decompress {zst_file}: {e}")
            if out_path.exists():
                out_path.unlink()  # Remove partial file
            continue
    
    print(f"  Decompression complete: {len(decompressed_files)} files, {total_samples} total samples")
    return decompressed_files


def _pick_text_column(ds) -> str:
    """Find best-effort text column name in a dataset."""
    candidates = ["text", "content", "document", "raw", "data"]
    for c in candidates:
        if c in ds.column_names:
            return c
    if ds.column_names:
        return ds.column_names[0]
    raise ValueError("Dataset has no columns; cannot locate text field.")


def _ensure_text_column(ds):
    """Ensure dataset has a 'text' column."""
    if "text" in ds.column_names:
        return ds
    col = _pick_text_column(ds)
    if col != "text":
        print(f"  Renaming column '{col}' to 'text'")
        ds = ds.rename_column(col, "text")
    return ds


def load_local_c4_validation(
    c4_dir: Path, 
    cache_dir: str,
    max_samples: Optional[int] = None
) -> 'datasets.Dataset':
    """
    Load local C4 validation data from c4-validation folder.
    
    Args:
        c4_dir: Path to local C4 validation directory
        cache_dir: Cache directory for HuggingFace
        max_samples: Optional limit on number of samples
        
    Returns:
        Loaded Dataset with 'text' column
    """
    from datasets import load_dataset, concatenate_datasets
    
    print(f"Loading C4 validation data from local directory: {c4_dir}")
    
    files = _find_c4_validation_files(c4_dir)
    n_files = sum(len(v) for v in files.values())
    
    if n_files == 0:
        raise RuntimeError(f"No supported files found under: {c4_dir}\n"
                          f"Supported extensions: .parquet, .jsonl, .json, .jsonl.zst, .json.gz")
    
    print("Found local C4 validation files:")
    for ext, lst in files.items():
        if lst:
            print(f"  {ext}: {len(lst)} files")
            for x in lst[:3]:
                print(f"    - {x}")
            if len(lst) > 3:
                print(f"    ... ({len(lst)-3} more)")
    
    dsets = []
    
    # Load parquet files
    if files["parquet"]:
        print(f"\nLoading {len(files['parquet'])} parquet files...")
        try:
            ds_parquet = load_dataset(
                "parquet",
                data_files=files["parquet"],
                split="train",
                cache_dir=cache_dir,
            )
            ds_parquet = _ensure_text_column(ds_parquet)
            dsets.append(ds_parquet)
            print(f"  Loaded {len(ds_parquet)} samples from parquet files")
        except Exception as e:
            print(f"  Warning: Failed to load parquet files: {e}")
    
    # Load jsonl/json files
    json_like = []
    json_like.extend(files["jsonl"])
    json_like.extend(files["json"])
    if json_like:
        print(f"\nLoading {len(json_like)} json/jsonl files...")
        try:
            ds_json = load_dataset(
                "json",
                data_files=json_like,
                split="train",
                cache_dir=cache_dir,
            )
            ds_json = _ensure_text_column(ds_json)
            dsets.append(ds_json)
            print(f"  Loaded {len(ds_json)} samples from json/jsonl files")
        except Exception as e:
            print(f"  Warning: Failed to load json/jsonl files: {e}")
    
    # Load .json.gz files (compressed JSON)
    if files.get("gz"):
        print(f"\nLoading {len(files['gz'])} gzip compressed json files...")
        try:
            ds_gz = load_dataset(
                "json",
                data_files=files["gz"],
                split="train",
                cache_dir=cache_dir,
            )
            ds_gz = _ensure_text_column(ds_gz)
            dsets.append(ds_gz)
            print(f"  Loaded {len(ds_gz)} samples from gzip files")
        except Exception as e:
            print(f"  Warning: Failed to load gzip files: {e}")
    
    # Load zst files using streaming
    if files["zst"]:
        print(f"\nLoading {len(files['zst'])} zstd compressed jsonl files...")
        try:
            import zstandard as zstd
            
            all_records = []
            for zst_file in files["zst"]:
                dctx = zstd.ZstdDecompressor()
                with open(zst_file, 'rb') as ifh:
                    with dctx.stream_reader(ifh) as reader:
                        text_stream = reader.read().decode('utf-8')
                        for line in text_stream.strip().split('\n'):
                            if line.strip():
                                try:
                                    record = json.loads(line)
                                    all_records.append(record)
                                except json.JSONDecodeError:
                                    continue
            
            if all_records:
                from datasets import Dataset
                ds_zst = Dataset.from_list(all_records)
                ds_zst = _ensure_text_column(ds_zst)
                dsets.append(ds_zst)
                print(f"  Loaded {len(ds_zst)} samples from zst files")
        except Exception as e:
            print(f"  Warning: Failed to load zst files: {e}")
    
    if not dsets:
        raise RuntimeError("Failed to load any datasets from local C4 validation files.")
    
    # Concatenate all loaded datasets
    if len(dsets) == 1:
        combined_ds = dsets[0]
    else:
        print(f"\nConcatenating {len(dsets)} datasets...")
        combined_ds = concatenate_datasets(dsets)
    
    print(f"\nTotal C4 validation samples loaded: {len(combined_ds)}")
    
    # Limit samples if requested
    if max_samples is not None and max_samples < len(combined_ds):
        print(f"Limiting to {max_samples} samples (from {len(combined_ds)})")
        combined_ds = combined_ds.select(range(max_samples))
    
    return combined_ds


def load_dclm_from_decompressed(
    dclm_output_dir: Path,
    cache_dir: str,
    max_samples: Optional[int] = None
) -> 'datasets.Dataset':
    """
    Load DCLM data from decompressed .jsonl files in DCLM folder.
    Supports loading from subdirectories.
    
    Uses batched loading strategy to handle Schema Inconsistency:
    - Load files in batches of 50
    - Keep only 'text' column immediately after loading each batch
    - This avoids PyArrow schema conflicts from inconsistent metadata fields
    - Fallback to individual file loading if a batch fails
    
    Args:
        dclm_output_dir: Path to DCLM folder with decompressed files
        cache_dir: Cache directory for HuggingFace
        max_samples: Optional limit on number of samples
        
    Returns:
        Loaded Dataset with 'text' column
    """
    from datasets import load_dataset, concatenate_datasets
    
    print(f"Loading DCLM data from decompressed files: {dclm_output_dir}")
    
    # Use rglob to find all .jsonl files including in subdirectories
    jsonl_files = sorted([str(p) for p in dclm_output_dir.rglob("*.jsonl")])
    
    if not jsonl_files:
        raise RuntimeError(f"No .jsonl files found in {dclm_output_dir} or its subdirectories")
    
    # Count subdirectories containing jsonl files
    subdirs = set()
    for f in jsonl_files:
        rel_path = Path(f).relative_to(dclm_output_dir)
        if len(rel_path.parts) > 1:
            subdirs.add(rel_path.parts[0])
    
    print(f"  Found {len(jsonl_files)} .jsonl files across {len(subdirs) if subdirs else 1} directories")
    
    if subdirs:
        print(f"  Subdirectories: {sorted(subdirs)[:5]}{'...' if len(subdirs) > 5 else ''}")
    
    # Show sample files
    for f in jsonl_files[:5]:
        rel_path = Path(f).relative_to(dclm_output_dir)
        print(f"    - {rel_path}")
    if len(jsonl_files) > 5:
        print(f"    ... ({len(jsonl_files) - 5} more)")
    
    # =========================================================================
    # ROBUST LOADING STRATEGY: Batched loading with column selection
    # =========================================================================
    # Why this approach?
    # 1. DCLM files have inconsistent metadata schemas across files
    #    (e.g., 'metadata' field can be null, int, string, or struct)
    # 2. PyArrow infers schema from first files and fails when later files differ
    # 3. Solution: Load in small batches, keep only 'text' column immediately
    #    This discards problematic metadata columns before concatenation
    # =========================================================================
    
    batch_size = 50  # Number of files per batch
    batches = [jsonl_files[i:i + batch_size] for i in range(0, len(jsonl_files), batch_size)]
    
    print(f"\n  Using robust batched loading strategy:")
    print(f"    - Batch size: {batch_size} files")
    print(f"    - Total batches: {len(batches)}")
    print(f"    - Strategy: Load batch -> Keep only 'text' column -> Concatenate")
    print(f"    - This handles schema inconsistencies in metadata fields")
    
    dsets = []
    total_loaded = 0
    failed_files = []
    
    for batch_idx, batch_files in enumerate(batches):
        # Check if we've reached max_samples
        if max_samples is not None and total_loaded >= max_samples:
            print(f"\n  Reached max_samples limit ({max_samples}), stopping.")
            break
        
        batch_num = batch_idx + 1
        
        try:
            # Load this batch of files
            ds_batch = load_dataset(
                "json",
                data_files=batch_files,
                split="train",
                cache_dir=cache_dir,
                keep_in_memory=False,  # Memory efficient
            )
            
            # Ensure we have a 'text' column
            ds_batch = _ensure_text_column(ds_batch)
            
            # CRITICAL: Keep ONLY the 'text' column
            # This discards all metadata columns (meta, url, timestamp, score, etc.)
            # which often have inconsistent types across files, causing:
            # "An error occurred while generating the dataset"
            if "text" in ds_batch.column_names:
                ds_batch = ds_batch.select_columns(["text"])
            
            dsets.append(ds_batch)
            total_loaded += len(ds_batch)
            
            # Progress report every 10 batches or at the end
            if batch_num % 10 == 0 or batch_num == len(batches):
                print(f"    Batch {batch_num}/{len(batches)}: +{len(ds_batch)} samples (Total: {total_loaded:,})")
        
        except Exception as e:
            print(f"\n    ⚠️  Batch {batch_num} failed: {e}")
            print(f"    Attempting individual file loading for this batch...")
            
            # Fallback: Load files individually from this batch
            batch_recovered = 0
            for file_path in batch_files:
                if max_samples is not None and total_loaded >= max_samples:
                    break
                
                try:
                    ds_single = load_dataset(
                        "json",
                        data_files=[file_path],
                        split="train",
                        cache_dir=cache_dir,
                        keep_in_memory=False,
                    )
                    ds_single = _ensure_text_column(ds_single)
                    
                    # Keep only 'text' column
                    if "text" in ds_single.column_names:
                        ds_single = ds_single.select_columns(["text"])
                    
                    dsets.append(ds_single)
                    total_loaded += len(ds_single)
                    batch_recovered += len(ds_single)
                    
                except Exception as inner_e:
                    file_name = Path(file_path).name
                    failed_files.append((file_path, str(inner_e)))
                    print(f"      ✗ Failed: {file_name}: {inner_e}")
            
            if batch_recovered > 0:
                print(f"    Recovered {batch_recovered} samples from batch {batch_num}")
    
    # Report failed files
    if failed_files:
        print(f"\n  ⚠️  {len(failed_files)} files failed to load:")
        for fp, err in failed_files[:5]:
            print(f"      - {Path(fp).name}: {err[:50]}...")
        if len(failed_files) > 5:
            print(f"      ... and {len(failed_files) - 5} more")
    
    if not dsets:
        raise RuntimeError("Failed to load any DCLM datasets. Check file formats and permissions.")
    
    # Concatenate all loaded datasets
    print(f"\n  Concatenating {len(dsets)} dataset batches...")
    
    if len(dsets) == 1:
        ds = dsets[0]
    else:
        ds = concatenate_datasets(dsets)
    
    print(f"  ✓ Total loaded: {len(ds):,} samples")
    
    # Limit samples if requested
    if max_samples is not None and max_samples < len(ds):
        print(f"  Limiting to {max_samples:,} samples (from {len(ds):,})")
        ds = ds.select(range(max_samples))
    
    # Clean up memory
    gc.collect()
    
    return ds


def load_llama2_tokenizer(cache_dir: str, max_length: int):
    """Load Llama-2-7b-hf tokenizer."""
    from transformers import AutoTokenizer
    
    print("Loading tokenizer: meta-llama/Llama-2-7b-hf")
    print("  (Requires HF_TOKEN with access to Meta-Llama models)")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            cache_dir=cache_dir,
            use_fast=True,
            model_max_length=max_length,
        )
    except Exception as e:
        print("\n" + "="*60)
        print("ERROR: Failed to load tokenizer meta-llama/Llama-2-7b-hf")
        print("="*60)
        print(f"Reason: {str(e)}")
        print("\nTroubleshooting (Windows):")
        print('  PowerShell: $env:HF_TOKEN="your_token"')
        print('  CMD: set HF_TOKEN=your_token')
        print("\nTroubleshooting (Linux):")
        print('  Bash: export HF_TOKEN="your_token"')
        print("\nSteps:")
        print("  1. Get a HuggingFace token at: https://huggingface.co/settings/tokens")
        print("  2. Request access at: https://huggingface.co/meta-llama/Llama-2-7b-hf")
        print("  3. Set your token before running this script")
        print("="*60)
        raise
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
        print(f"  Set pad_token to unk_token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    
    return tokenizer


def tokenize_dataset(
    dataset,
    tokenizer,
    max_length: int,
    num_proc: int,
    desc: str,
    cache_dir: str
):
    """Tokenize a dataset using CPU parallelism."""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
    
    print(f"\nTokenizing {desc} with {num_proc} CPU processes...")
    print(f"  Max length: {max_length}")
    print(f"  Samples: {len(dataset)}")
    
    start_time = time.time()
    
    cache_file_name = os.path.join(
        cache_dir, 
        f"tokenized_cache_{desc.lower().replace(' ', '_')}_{len(dataset)}_{max_length}.arrow"
    )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {desc}",
        cache_file_name=cache_file_name,
    )
    
    elapsed = time.time() - start_time
    samples_per_sec = len(dataset) / elapsed if elapsed > 0 else 0
    
    print(f"  Completed in {elapsed:.1f}s ({samples_per_sec:.0f} samples/sec)")
    print(f"  Output features: {list(tokenized.features.keys())}")
    
    return tokenized


def verify_tokenized_dataset(ds, name: str) -> bool:
    """Verify that a tokenized dataset has required fields."""
    required_keys = ["input_ids", "attention_mask"]
    missing = [k for k in required_keys if k not in ds.features]
    
    if missing:
        raise ValueError(f"Tokenized {name} dataset missing required keys: {missing}")
    
    if len(ds) > 0:
        sample = ds[0]
        for key in required_keys:
            if len(sample[key]) == 0:
                raise ValueError(f"Tokenized {name} has empty {key}")
    
    return True


def save_dataset_info(
    output_dir: Path,
    train_path: str,
    eval_path: str,
    num_train: int,
    num_eval: int,
    max_length: int,
    tokenizer,
    dclm_dir: str,
    c4_dir: str
) -> str:
    """Save dataset_info_llama2.json compatible with train_Llama2.py."""
    dataset_info = {
        "train_path": train_path,
        "eval_path": eval_path,
        "num_train_samples": num_train,
        "num_eval_samples": num_eval,
        "max_length": max_length,
        "tokenizer": "meta-llama/Llama-2-7b-hf",
        "vocab_size": tokenizer.vocab_size,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "dataset": "dclm_baseline_local",
        "validation_dataset": "c4_en_local",
        "training_source": f"local_files:{dclm_dir}",
        "validation_source": f"local_files:{c4_dir}",
        "created_at": datetime.now().isoformat(),
    }

    info_path = output_dir / "dataset_info_llama2.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    return str(info_path)


def main() -> int:
    """Main function."""
    args = parse_args()

    print("=" * 80)
    print("Prepare local DCLM shard + C4/en validation (Llama-2 tokenizer)")
    print("=" * 80)
    
    cache_paths = setup_hf_cache_dirs(args.cache_dir)

    # Output directory is now 'data' by default
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # DCLM output directory for decompressed files
    dclm_output_dir = Path(args.dclm_output_dir)
    dclm_output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir_abs = os.path.abspath(os.path.join(os.getcwd(), args.cache_dir))
    dclm_dir = Path(args.dclm_dir)
    c4_dir = Path(args.c4_validation_dir)

    print(f"DCLM source dir      : {dclm_dir.resolve()}")
    print(f"DCLM decompressed dir: {dclm_output_dir.resolve()}")
    print(f"C4 validation dir    : {c4_dir.resolve()}")
    print(f"Output directory     : {out_dir.resolve()}")
    print(f"Cache directory      : {cache_dir_abs}")
    print(f"Max length           : {args.max_length}")
    print(f"CPU parallelism      : {args.num_proc}")
    print(f"Batch size           : {args.batch_size}")
    print(f"Shuffle train        : {args.shuffle_train}")
    print(f"DCLM max samples     : {args.dclm_max_samples or 'all'}")
    print(f"C4 max samples       : {args.c4_max_samples or 'all (full validation)'}")
    print("\nHuggingFace cache paths:")
    for k, v in cache_paths.items():
        print(f"  {k}: {v}")
    print("=" * 80)

    try:
        import zstandard
        print("\n✓ zstandard library is installed")
    except ImportError:
        print("\n⚠️  WARNING: zstandard library not found")
        print("  Run: pip install zstandard")
        return 1

    # ============================================================
    # Step 1/4: Load Llama-2 tokenizer
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 1/4: Loading Llama-2 tokenizer")
    print("=" * 60)
    
    try:
        tokenizer = load_llama2_tokenizer(cache_dir_abs, args.max_length)
    except Exception as e:
        print(f"\nFailed to load tokenizer: {e}")
        return 1

    print("\nTokenizer configuration:")
    print(f"  vocab_size   : {tokenizer.vocab_size}")
    print(f"  bos_token_id : {tokenizer.bos_token_id}")
    print(f"  eos_token_id : {tokenizer.eos_token_id}")
    print(f"  pad_token_id : {tokenizer.pad_token_id}")
    print(f"  unk_token_id : {tokenizer.unk_token_id}")
    
    if tokenizer.vocab_size != 32000:
        print(f"\n⚠️  WARNING: Expected vocab_size=32000, got {tokenizer.vocab_size}")
    else:
        print(f"\n✓ Tokenizer vocab_size matches Llama 2 (32000)")

    # ============================================================
    # Step 2/4: Load and process local C4 validation data
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 2/4: Loading local C4 validation data from c4-validation folder")
    print("=" * 60)
    
    try:
        c4_val = load_local_c4_validation(c4_dir, cache_dir_abs, args.c4_max_samples)
    except Exception as e:
        print(f"\nFailed to load C4 validation data: {e}")
        return 1

    print(f"\nC4/en validation dataset:")
    print(f"  Samples: {len(c4_val)}")
    print(f"  Columns: {c4_val.column_names}")

    # Tokenize C4 validation data
    tokenized_eval = tokenize_dataset(
        c4_val,
        tokenizer,
        args.max_length,
        args.num_proc,
        "C4 validation",
        cache_dir_abs
    )

    # ============================================================
    # Step 3/4: Decompress DCLM raw data to DCLM folder (with caching)
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 3/4: Decompressing DCLM raw data to DCLM folder")
    print("=" * 60)
    
    # Check if DCLM folder already has decompressed files
    if _check_dclm_decompressed(dclm_output_dir):
        print(f"\n✓ DCLM folder already contains decompressed files. Skipping decompression.")
    else:
        print(f"\n  DCLM folder is empty or does not exist. Starting decompression...")
        
        # Find .zst files in source directory
        files = _find_local_data_files(dclm_dir)
        
        if files["zst"]:
            print(f"  Found {len(files['zst'])} .zst files to decompress")
            
            try:
                _decompress_zst_to_folder(
                    files["zst"],
                    dclm_dir,  # Pass source root for relative path calculation
                    dclm_output_dir, 
                    args.dclm_max_samples
                )
                print(f"\n✓ Decompression complete. Files saved to {dclm_output_dir}")
            except Exception as e:
                print(f"\nFailed to decompress DCLM data: {e}")
                return 1
        else:
            # If no .zst files, check for other formats and copy/use directly
            print(f"  No .zst files found. Checking for other formats...")
            
            if files["jsonl"] or files["json"] or files["parquet"]:
                print(f"  Found other file formats:")
                for ext, lst in files.items():
                    if lst and ext != "zst":
                        print(f"    {ext}: {len(lst)} files")
                print(f"  These will be loaded directly in Step 4.")
            else:
                print(f"\n⚠️  WARNING: No supported files found in {dclm_dir}")
                return 1

    # ============================================================
    # Step 4/4: Load and process DCLM data
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 4/4: Processing DCLM dataset with 8 parallel workers")
    print("=" * 60)
    
    try:
        # First try to load from decompressed DCLM folder
        if _check_dclm_decompressed(dclm_output_dir):
            dclm_train_raw = load_dclm_from_decompressed(
                dclm_output_dir, 
                cache_dir_abs, 
                args.dclm_max_samples
            )
        else:
            # Fall back to loading directly from source (for non-.zst files)
            from datasets import load_dataset, concatenate_datasets
            
            files = _find_local_data_files(dclm_dir)
            dsets = []
            
            # Load parquet files
            if files["parquet"]:
                print(f"\nLoading {len(files['parquet'])} parquet files...")
                ds_parquet = load_dataset(
                    "parquet",
                    data_files=files["parquet"],
                    split="train",
                    cache_dir=cache_dir_abs,
                )
                ds_parquet = _ensure_text_column(ds_parquet)
                dsets.append(ds_parquet)
                print(f"  Loaded {len(ds_parquet)} samples from parquet files")
            
            # Load jsonl/json files
            json_like = []
            json_like.extend(files["jsonl"])
            json_like.extend(files["json"])
            if json_like:
                print(f"\nLoading {len(json_like)} json/jsonl files...")
                ds_json = load_dataset(
                    "json",
                    data_files=json_like,
                    split="train",
                    cache_dir=cache_dir_abs,
                )
                ds_json = _ensure_text_column(ds_json)
                dsets.append(ds_json)
                print(f"  Loaded {len(ds_json)} samples from json/jsonl files")
            
            if not dsets:
                raise RuntimeError("No data loaded from DCLM source")
            
            if len(dsets) == 1:
                dclm_train_raw = dsets[0]
            else:
                dclm_train_raw = concatenate_datasets(dsets)
            
            # Limit samples if requested
            if args.dclm_max_samples and args.dclm_max_samples < len(dclm_train_raw):
                dclm_train_raw = dclm_train_raw.select(range(args.dclm_max_samples))
    
    except Exception as e:
        print(f"\nFailed to load DCLM data: {e}")
        return 1

    if args.shuffle_train:
        print("\nShuffling training dataset...")
        dclm_train_raw = dclm_train_raw.shuffle(seed=args.seed)

    print(f"\nLocal DCLM dataset:")
    print(f"  Samples: {len(dclm_train_raw)}")
    print(f"  Columns: {dclm_train_raw.column_names}")

    # Tokenize DCLM training data
    tokenized_train = tokenize_dataset(
        dclm_train_raw,
        tokenizer,
        args.max_length,
        args.num_proc,
        "DCLM train",
        cache_dir_abs
    )

    # ============================================================
    # Verification and Saving
    # ============================================================
    print("\n" + "=" * 60)
    print("Verifying and saving tokenized datasets")
    print("=" * 60)

    print("\nVerifying tokenized datasets...")
    try:
        verify_tokenized_dataset(tokenized_train, "train")
        verify_tokenized_dataset(tokenized_eval, "eval")
        print("  ✓ All datasets have required fields (input_ids, attention_mask)")
    except ValueError as e:
        print(f"\nDataset verification failed: {e}")
        return 1

    # Save to data folder
    train_out = out_dir / "dclm_local_train_llama2"
    eval_out = out_dir / "c4_en_validation_llama2"

    print(f"\nSaving tokenized train dataset to: {train_out}")
    tokenized_train.save_to_disk(str(train_out))

    print(f"Saving tokenized eval dataset to: {eval_out}")
    tokenized_eval.save_to_disk(str(eval_out))

    info_path = save_dataset_info(
        out_dir,
        str(train_out),
        str(eval_out),
        len(tokenized_train),
        len(tokenized_eval),
        args.max_length,
        tokenizer,
        str(dclm_dir.resolve()),
        str(c4_dir.resolve())
    )

    print("\n" + "=" * 80)
    print("✓ Dataset preparation completed successfully!")
    print("=" * 80)
    print(f"\nOutputs (all in {out_dir} folder):")
    print(f"  Train dataset   : {train_out}")
    print(f"    Samples       : {len(tokenized_train)}")
    print(f"  Eval dataset    : {eval_out}")
    print(f"    Samples       : {len(tokenized_eval)}")
    print(f"  Dataset info    : {info_path}")
    
    print("\n" + "-" * 80)
    print("Training command example:")
    print("-" * 80)
    print(f"""
torchrun --standalone --nproc_per_node 8 train_Llama2.py \\
    --model_config configs/llama_1.2b.json \\
    --lr 4e-3 \\
    --adamw_lr_multiplier 0.3 \\
    --batch_size 16 \\
    --total_batch_size 256 \\
    --max_length {args.max_length} \\
    --activation_checkpointing \\
    --num_training_steps 11500 \\
    --warmup_steps 0 \\
    --stable_steps 0 \\
    --weight_decay 0.1 \\
    --grad_clipping 2.0 \\
    --betas 0.8 0.98 \\
    --dtype bfloat16 \\
    --eval_every 100 \\
    --scheduler linear \\
    --min_lr_ratio 0 \\
    --optimizer muon \\
    --save_dir checkpoints/llama_1.2b_4096seq \\
    --name llama_1.2b_muon \\
    --wandb_project "stf-igcai-1.2b" \\
    --wandb_name "muon" \\
    --dataset_info {info_path}
""")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())