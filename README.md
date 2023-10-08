# chatglm2-BigDL
通过intel发布的bigDL,可以用cpu运行chatglm2,
操作步骤：
1.conda create -n inthj python==3.9
2.conda activate inthj
3.cd inthj
4.pip install --pre --upgrade bigdl-llm[all]
5.进入chatglm2的cli_demo.py文件
6.把from transformers import AutoTokenizer, AutoModel
改成from bigdl.llm.transformers import AutoModel
from transformers import AutoTokenizer
7.把model = AutoModel.from_pretrained("你的路径",trust_remote_code=True).float()
改成model = AutoModel.from_pretrained("你的路径", load_in_4bit=True,trust_remote_code=True).float()
8.然后你就看见你的大模型在CPU下飞快的运转了
