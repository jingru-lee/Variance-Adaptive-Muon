import os
from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path

def download_dclm_shards():
    """
    下载 mlfoundations/dclm-baseline-1.0 的第一个 global-shard 中的前两个 local-shards
    """
    repo_id = "mlfoundations/dclm-baseline-1.0"
    
    # 设置缓存目录为当前目录
    cache_dir = os.path.join(os.getcwd(), ".cache")
    download_dir = os.getcwd()
    
    print(f"缓存目录: {cache_dir}")
    print(f"下载目录: {download_dir}")
    
    # 列出所有文件
    print("正在获取文件列表...")
    all_files = list_repo_files(repo_id, repo_type="dataset")
    
    # 筛选出第一个 global-shard 中的前两个 local-shards 的文件
    # 通常命名格式为: global-shard_XX/local-shard_YY/...
    target_files = []
    for file in all_files:
        parts = file.split('/')
        if len(parts) >= 2:
            # 检查是否是第一个 global-shard (global-shard_00 或 global-shard_0)
            if parts[0] in ['global-shard_00', 'global-shard_0', 'global-shard_000']:
                # 检查是否是前两个 local-shard
                if len(parts) >= 2 and parts[1] in ['local-shard_00', 'local-shard_0', 
                                                      'local-shard_01', 'local-shard_1',
                                                      'local-shard_000', 'local-shard_001']:
                    target_files.append(file)
    
    if not target_files:
        print("未找到匹配的文件，尝试其他命名模式...")
        # 尝试其他可能的命名模式
        for file in all_files:
            if 'global' in file.lower() and 'local' in file.lower():
                parts = file.split('/')
                if len(parts) >= 1:
                    target_files.append(file)
    
    print(f"\n找到 {len(target_files)} 个文件需要下载:")
    for f in target_files[:10]:  # 只显示前10个
        print(f"  - {f}")
    if len(target_files) > 10:
        print(f"  ... 还有 {len(target_files) - 10} 个文件")
    
    # 下载文件
    for i, file_path in enumerate(target_files, 1):
        print(f"\n[{i}/{len(target_files)}] 正在下载: {file_path}")
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type="dataset",
                cache_dir=cache_dir,
                local_dir=download_dir,
                local_dir_use_symlinks=False  # 直接复制文件，不使用符号链接
            )
            print(f"已下载到: {downloaded_path}")
        except Exception as e:
            print(f"下载失败: {e}")
            continue
    
    print("\n下载完成!")
    print(f"所有文件已保存到: {download_dir}")

if __name__ == "__main__":
    download_dclm_shards()