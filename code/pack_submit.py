# -*- coding: utf-8 -*-
# 打包提交：去空格 -> 校验 -> 压缩为 zip（内含 pred_results.csv）
import os
import zipfile

OUT = r"D:\aicomp\output"
csv_path = os.path.join(OUT, "pred_results.csv")

# 1) 去掉 "逗号+空格" 中的空格（文件名是hex哈希不含空格，编号四位纯数字，安全）
with open(csv_path, "rb") as f:
    data = f.read()
data = data.replace(b", ", b",")
with open(csv_path, "wb") as f:
    f.write(data)

lines = data.decode("utf-8").strip().splitlines()
print("rows:", len(lines))
print("first 3:", lines[:3])
print("last  1:", lines[-1:])

bad = [l for l in lines if l.count(",") != 1 or len(l.split(",")[1].strip()) != 4]
print("bad rows:", len(bad), bad[:5])

# 2) 压缩，zip 内文件名固定为 pred_results.csv
zip_path = os.path.join(OUT, "pred_results.zip")
if os.path.exists(zip_path):
    os.remove(zip_path)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(csv_path, arcname="pred_results.csv")
print("zip:", zip_path, "| size:", os.path.getsize(zip_path), "bytes")
