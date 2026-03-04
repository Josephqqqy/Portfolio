Django + LangChain + FAISS 聊天机器人示例（支持多轮对话）
---------------------------------------

## 功能特性
- ✅ RAG（检索增强生成）问答系统
- ✅ 多轮上下文对话功能
- ✅ 自动提取联系方式（手机、微信、QQ等）
- ✅ 负面反馈检测
- ✅ 会话历史持久化

## 快速开始

### 1. 环境配置
```bash
# 将 .env.template 复制为 .env 并填写 OPENAI_API_KEY
cp .env.template .env
# 编辑 .env 文件，添加你的 OpenAI API Key
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 数据库迁移
```bash
python manage.py migrate
```

### 4. 准备知识库
```bash
# 将你的文档（txt格式）放到 data/docs/ 目录
# 然后构建FAISS索引
python manage.py build_faiss_index
```

### 5. 启动服务
```bash
python manage.py runserver
```

访问 http://localhost:8000/ 开始对话！

## 多轮对话功能

系统会自动记住对话历史，你可以：
- 使用代词（"它"、"那个"等）引用之前提到的内容
- 进行连续追问，无需重复上下文
- 点击"清空对话"按钮开始新会话

示例对话：
```
用户: 你们的营业时间是什么？
AI: [回答营业时间]

用户: 周末也营业吗？
AI: [理解"周末"是在问营业时间，结合上下文回答]
```

## 管理命令

### 构建FAISS索引
```bash
python manage.py build_faiss_index
```

### 清理旧会话
```bash
# 删除30天前的会话
python manage.py cleanup_old_sessions --days 30

# 模拟运行（不实际删除）
python manage.py cleanup_old_sessions --days 30 --dry-run
```

## 测试

运行测试脚本验证多轮对话功能：
```bash
python test_multiround_chat.py
```

## 技术栈
- Django 4.2+
- LangChain
- FAISS
- OpenAI API
- SQLite（可更换为其他数据库）

详细说明请查看 `多轮对话功能说明.md`
