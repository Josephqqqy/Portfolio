"""
This script creates a CLI demo with transformers backend for the glm-4-9b model,
allowing users to interact with the model through a command-line interface.

Usage:
- Run the script to start the CLI demo.
- Interact with the model by typing questions and receiving responses.

Note: The script includes a modification to handle markdown to plain text conversion,
ensuring that the CLI interface displays formatted text correctly.

If you use flash attention, you should install the flash-attn and  add attn_implementation="flash_attention_2" in model loading.
"""

import os  # 导入os库，用于处理环境变量和文件路径
import torch  # 导入torch库，用于深度学习计算
from threading import Thread  # 导入Thread用于多线程
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, AutoModel  # 导入transformers相关类
import tornado.web  # 导入tornado的web模块
import tornado.ioloop  # 导入tornado的ioloop事件循环
from tornado.web import RequestHandler  # 导入RequestHandler用于处理http请求
MODEL_PATH = os.environ.get('MODEL_PATH', '/root/autodl-tmp/glm-4-9b-chat')  # 获取模型路径，若环境变量未设置则使用默认路径

## 如果使用peft微调模型，可以取消以下注释
# def load_model_and_tokenizer(model_dir, trust_remote_code: bool = True):
#     if (model_dir / 'adapter_config.json').exists():  # 若适配器配置文件存在，加载peft模型
#         model = AutoModel.from_pretrained(
#             model_dir, trust_remote_code=trust_remote_code, device_map='auto'
#         )
#         tokenizer_dir = model.peft_config['default'].base_model_name_or_path  # 获取基础模型路径
#     else:
#         model = AutoModel.from_pretrained(
#             model_dir, trust_remote_code=trust_remote_code, device_map='auto'
#         )
#         tokenizer_dir = model_dir  # 否则tokenizer使用当前目录
#     tokenizer = AutoTokenizer.from_pretrained(
#         tokenizer_dir, trust_remote_code=trust_remote_code, use_fast=False
#     )
#     return model, tokenizer  # 返回模型和分词器


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,  # 使用指定路径加载分词器
    trust_remote_code=True,  # 信任远程代码
    encode_special_tokens=True  # 编码特殊token
)

model = AutoModel.from_pretrained(
    MODEL_PATH,  # 使用指定路径加载模型
    trust_remote_code=True,  # 信任远程代码
    # attn_implementation="flash_attention_2", # 若使用flash-attn可取消注释
    # torch_dtype=torch.bfloat16, # 使用flash-attn需要bfloat16或float16
    device_map="auto"  # 自动分配到可用设备
).eval()  # 设置为评估模式


class StopOnTokens(StoppingCriteria):  # 自定义停止条件类
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = model.config.eos_token_id  # 获取eos（结束）token id
        for stop_id in stop_ids:  # 遍历所有eos token id
            if input_ids[0][-1] == stop_id:  # 如果当前生成的最后一个token是结束符
                return True  # 返回True停止生成
        return False  # 否则继续生成

history = []  # 创建全局对话历史
def chatbot_api(user_input):
    max_length = 8192  # 生成最大token数量
    top_p = 0.8  # top-p采样率
    temperature = 0.6  # 温度采样参数
    stop = StopOnTokens()  # 创建自定义停止条件

    print("Welcome to the GLM-4-9B CLI chat. Type your messages below.")  # 打印欢迎语

    history.append([user_input, ""])  # 将用户输入和空回复加入历史记录

    messages = []  # 初始化消息列表
    for idx, (user_msg, model_msg) in enumerate(history):  # 遍历历史
        if idx == len(history) - 1 and not model_msg:  # 最后一条且模型回复为空
            messages.append({"role": "user", "content": user_msg})  # 仅添加用户输入消息
            break  # 跳出循环
        if user_msg:  # 如果有用户消息
            messages.append({"role": "user", "content": user_msg})  # 加入用户消息
        if model_msg:  # 如果有助手回复
            messages.append({"role": "assistant", "content": model_msg})  # 加入助手回复
    model_inputs = tokenizer.apply_chat_template(
        messages,  # 输入聊天历史
        add_generation_prompt=True,  # 添加生成提示符
        tokenize=True,  # 是否分词
        return_tensors="pt"  # 返回pytorch张量
    ).to(model.device)  # 移动到模型设备

    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,  # 使用的分词器
        timeout=60,  # 超时时间
        skip_prompt=True,  # 跳过原始输入
        skip_special_tokens=True  # 跳过特殊token
    )
    generate_kwargs = {  # 生成参数
        "input_ids": model_inputs,  # 输入张量
        "streamer": streamer,  # 用于流式输出
        "max_new_tokens": max_length,  # 生成最大token数
        "do_sample": True,  # 为True表示启用采样生成（否则为贪婪搜索）；采样能提升回复多样性
        "top_p": top_p,  # top-p采样
        "temperature": temperature,  # 温度
        "stopping_criteria": StoppingCriteriaList([stop]),  # 自定义停止条件
        "repetition_penalty": 1.2,  # 重复惩罚系数
        "eos_token_id": model.config.eos_token_id,  # 结束token id
    }
    t = Thread(target=model.generate, kwargs=generate_kwargs)  # 用多线程调用生成函数
    t.start()  # 启动线程
    print("GLM-4:", end="", flush=True)  # 打印前缀，准备输出回复
    for new_token in streamer:  # 流式获取新token
        if new_token:  # 如果token非空
            print(new_token, end="", flush=True)  # 输出到控制台
            history[-1][1] += new_token  # 将新token追加到最后一条历史回复

    history[-1][1] = history[-1][1].strip()  # 去除回复首尾空格
    return history[-1][1]  # 返回新回复

class BaseHandler(RequestHandler):
    """解决JS跨域请求问题"""

    def set_default_headers(self):  # 设置响应头允许跨域
        self.set_header('Access-Control-Allow-Origin', '*')  # 允许所有来源跨域
        self.set_header('Access-Control-Allow-Methods', 'POST, GET')  # 允许POST、GET
        self.set_header('Access-Control-Max-Age', 1000)  # 预检请求最大有效时间
        self.set_header('Access-Control-Allow-Headers', '*')  # 允许所有header
        # self.set_header('Content-type', 'application/json')  # 可选设置为json返回类型


class IndexHandler(BaseHandler):  # 主页接口处理
    # 添加一个处理get请求方式的方法
    def get(self):
        # 向响应中，添加数据
        infos = self.get_query_argument("infos")  # 获取前端参数infos
        print("Q:", infos)  # 打印用户问题
        # 捕捉服务器异常信息
        try:
            result = chatbot_api(user_input=infos)  # 调用主对话接口获取模型回复
        except Exception as e:  # 捕捉异常
            print(e)  # 打印错误
            result = "服务器内部错误"  # 返回错误信息
        print("A:", "".join(result))  # 打印模型回复
        self.write("".join(result)) #发送给前端

if __name__ == '__main__':  # 主程序入口
    # 创建一个应用对象
    app = tornado.web.Application([(r'/api/chatbot', IndexHandler)])  # 设置路由与处理类
    # 绑定一个监听端口
    app.listen(6006)  # 监听6006端口
    # 启动web程序，开始监听端口的连接
    tornado.ioloop.IOLoop.current().start()  # 启动主事件循环