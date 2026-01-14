import os
import zipfile
import urllib.request
import shutil
import sys
import subprocess

# FFmpeg 下载地址 (BtbN builds)
FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
DOWNLOAD_DIR = "tools"
FFMPEG_DIR = os.path.join(DOWNLOAD_DIR, "ffmpeg")

def download_and_setup_ffmpeg():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    zip_path = os.path.join(DOWNLOAD_DIR, "ffmpeg.zip")
    
    print(f"⬇️ Downloading FFmpeg from {FFMPEG_URL}...")
    try:
        # 使用 urllib 下载
        urllib.request.urlretrieve(FFMPEG_URL, zip_path)
        print("✅ Download complete.")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

    print("📦 Extracting FFmpeg...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DOWNLOAD_DIR)
        
        # 查找解压后的 bin 目录
        extracted_folder = next((d for d in os.listdir(DOWNLOAD_DIR) if d.startswith("ffmpeg-master")), None)
        if not extracted_folder:
            print("❌ Could not find extracted folder.")
            return False
            
        bin_path = os.path.join(DOWNLOAD_DIR, extracted_folder, "bin")
        
        # 移动 ffmpeg.exe 到 tools/ffmpeg
        if not os.path.exists(FFMPEG_DIR):
            os.makedirs(FFMPEG_DIR)
            
        shutil.copy(os.path.join(bin_path, "ffmpeg.exe"), os.path.join(FFMPEG_DIR, "ffmpeg.exe"))
        shutil.copy(os.path.join(bin_path, "ffprobe.exe"), os.path.join(FFMPEG_DIR, "ffprobe.exe"))
        
        print(f"✅ FFmpeg setup complete in {FFMPEG_DIR}")
        
        # 清理临时文件
        os.remove(zip_path)
        shutil.rmtree(os.path.join(DOWNLOAD_DIR, extracted_folder))
        
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

if __name__ == "__main__":
    if not os.path.exists(os.path.join(FFMPEG_DIR, "ffmpeg.exe")):
        download_and_setup_ffmpeg()
    else:
        print("✅ FFmpeg already exists.")