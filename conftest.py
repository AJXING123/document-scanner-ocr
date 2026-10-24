"""pytest 根配置：保证可以 import 仓库根目录下的 scanner 模块。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
