from __future__ import annotations
import os
import argparse
import numpy as np
from tokenizer import Tokenizer
from pathlib import Path



def read_documents(input_path: str) -> list[str]:
    p = Path(input_path)
    docs: list[str] = []
    if p.is_dir():  # 目录
        for file_path in p.glob("**/*.txt"):  # 递归查找所有txt文件
            with open(file_path, "r", encoding="utf-8") as f:
                docs.append(f.read())
    else:
        with open(p, "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs