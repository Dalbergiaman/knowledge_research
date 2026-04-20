from fastapi import UploadFile, File
import requests


url = "http://localhost:8000/insert/file"
file_path = "rag/test_doc/ai_dev.md"  # 你的 markdown 文件路径

# 打开文件并发送
with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "text/markdown")}
    response = requests.post(url, files=files)

print(response.json())