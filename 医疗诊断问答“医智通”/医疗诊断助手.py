from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
from peft import PeftModel
import pandas as pd

# 创建一个标题和一个副标题
st.title("💬 Yuan2.0 医疗诊断问答“医智通”")

# 源大模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./')

# 定义模型路径
path = './IEITYuan/Yuan2-2B-Mars-hf'
lora_path = './output/Yuan2.0-2B_lora_bf16/checkpoint-51'

# 定义模型数据类型
torch_dtype = torch.bfloat16  # A10
# torch_dtype = torch.float16  # P100

# 定义一个函数，用于获取模型和tokenizer
@st.cache_resource
def get_model():
    print("Creat tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
    tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>',
                         '<commit_before>', '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>',
                         '<jupyter_code>', '<jupyter_output>', '<empty_output>'], special_tokens=True)

    print("Creat model...")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype, trust_remote_code=True).cuda()
    model = PeftModel.from_pretrained(model, model_id=lora_path)

    return tokenizer, model

# 加载model和tokenizer
tokenizer, model = get_model()

template = '''
# 任务描述
假设你是一个AI医疗诊断助手，能够根据用户提供的症状描述给出相应的医疗建议。

# 任务要求
返回的医疗建议应该包括诊断、治疗建议、可能的原因以及预防措施。

# 样例
输入：
我家的孩子是男宝宝，9岁，刚开始，嗓子眼有点痛
输出：
如果孩子得了扁桃体炎的话，首先可以用点对症的抗生素药物，也可局部冲洗或是局部喷药，扁桃体内也可注射对症药物，疗效都是不错的，症状如果是以咽痛为主的，可考虑给点镇痛类的药物，另外如果伴有发烧的情况的话，那么也可服用一些退烧药，高烧的话还是建议要尽早就医的，出现多次急性严重，或是已有并发症的，建议急性炎症消退二周后施行扁桃体切除术，家长平时还要注意给孩子做好保暖工作，以免受凉感冒诱发扁桃体的再次发炎。

# 当前文本
input_str

# 任务重述
请参考样例，按照任务要求，根据当前文本中的症状描述给出医疗建议。
'''

# 在聊天界面上显示模型的输出
st.chat_message("assistant").write(f"请输入症状描述：")

# 如果用户在聊天输入框中输入了内容，则执行以下操作
if query := st.chat_input():
    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(query)

    # 调用模型
    prompt = template.replace('input_str', query).strip()
    prompt += "<sep>"
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    outputs = model.generate(inputs, do_sample=False, max_length=8192)  # 设置解码方式和最大生成长度
    output = tokenizer.decode(outputs[0])
    response = output.split("<sep>")[-1].replace("<eod>", '').strip()

    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(f"正在生成医疗建议，请稍候...")

    # 显示模型的输出
    st.chat_message("assistant").write(response)
