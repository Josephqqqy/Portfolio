# encoding: utf-8


import os
import tornado.web
import tornado.ioloop
from tornado.web import RequestHandler
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
# pip install torch --index-url https://download.pytorch.org/whl/cu124
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 设置 GPU 编号，如果单机单卡指定一个，单机多卡指定多个 GPU 编号
MODEL_PATH = "./chatglm4-9b"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
).eval()


def chatbot_api(infos,history):

    # 这两句是直接从huggingface下载权重
    # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    # model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).quantize(4).half().cuda()

    if history is None:
        history = []
    print(infos)
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": infos}],
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True
                                           )

    inputs = inputs.to(device)
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response=tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
    return response

#代码不用看
class BaseHandler(RequestHandler):
    """解决JS跨域请求问题"""

    def set_default_headers(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET')
        self.set_header('Access-Control-Max-Age', 1000)
        self.set_header('Access-Control-Allow-Headers', '*')
        # self.set_header('Content-type', 'application/json')


class IndexHandler(BaseHandler):
    # 添加一个处理get请求方式的方法
    def get(self):
        # 向响应中，添加数据
        infos = self.get_query_argument("infos")
        print("Q:", infos)
        # 捕捉服务器异常信息
        try:
            result = chatbot_api(infos=infos,history=None)#调用训练好的模型，预测得到结果，结果给了result
        except Exception as e:
            print(e)
            result = "服务器内部错误"
        print("A:", "".join(result))
        self.write("".join(result)) #发送给前端


if __name__ == '__main__':
    # 创建一个应用对象
    app = tornado.web.Application([(r'/api/chatbot', IndexHandler)])
    # 绑定一个监听端口
    app.listen(8000)
    # 启动web程序，开始监听端口的连接
    tornado.ioloop.IOLoop.current().start()
