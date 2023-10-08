# chatglm2-BigDL
#通过intel发布的bigDL,可以用cpu运行chatglm2,
#操作步骤：
#1.conda create -n inthj python==3.9
#2.conda activate inthj
#3.cd inthj
#4.pip install --pre --upgrade bigdl-llm[all]
#5.进入chatglm2的cli_demo.py文件
#6.把from transformers import AutoTokenizer, AutoModel
#改成from bigdl.llm.transformers import AutoModel
#from transformers import AutoTokenizer
#7.把model = AutoModel.from_pretrained("你的路径",trust_remote_code=True).float()
#改成model = AutoModel.from_pretrained("你的路径", load_in_4bit=True,trust_remote_code=True).float()
#8.然后你就看见你的大模型在CPU下飞快的运转了
import os
import platform
import signal
#from transformers import AutoTokenizer, AutoModel
#import readline
import torch
import time
import argparse
import numpy as np

from bigdl.llm.transformers import AutoModel
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("D:\\c-glm\\ChatGLM2-6B", trust_remote_code=True)
model = AutoModel.from_pretrained("D:\\c-glm\\ChatGLM2-6B", load_in_4bit=True,trust_remote_code=True)#.float()#.cuda()

# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
# from utils import load_model_on_gpus
# model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM2-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    past_key_values, history = None, []
    global stop_stream
    print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        print("\nChatGLM：", end="")
        current_length = 0
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        print("")


if __name__ == "__main__":
    main()

