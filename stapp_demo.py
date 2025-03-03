# 基础功能demo
# 集成了向量数据库Chroma的RAG知识库查询+自动发送邮件

import os
import shutil
import time
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
from openai import OpenAI
import streamlit as st
from llama_index.core import VectorStoreIndex, Settings,SimpleDirectoryReader,StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from fuzzywuzzy import process
import re
from config import settings
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chroma import query_with_RAG

GPT_MODEL = "gpt-4o-mini"

load_dotenv()
# 获取环境变量
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AUTHORIZATION_CODE = os.getenv("AUTHORIZATION_CODE")

# client = openai.OpenAI(api_key=OPENAI_API_KEY)
client = OpenAI(base_url="https://api.openai-proxy.org/v1",
                api_key=OPENAI_API_KEY)
tools = [
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to the specified email with the subject and content",
            "parameters": {
                "type": "object",
                "properties": {
                    "FromEmail": {
                        "type": "string",
                        "description": "The email address, eg., remember0101@126.com",
                    },
                    "Subject": {
                        "type": "string",
                        "description": "Subject of the email",
                    },
                    "Body": {
                        "type": "string",
                        "description": "The content of the email",
                    },
                    "Recipients": {
                        "type": "string",
                        "description": "The recipients' email addresses",
                    }
                },
                "required": ["FromEmail", "Subject", "Body", "Recipients"],
            },
        }
    }
]

st.sidebar.header("📃 Dialgue Session:")

# 定义 Prompt 库
KNOWLEDGE_BASE_KEYWORDS = [
    "知识库", "知识库查询", "文档", "资料", "信息",
    "notion", "Notion", "检索", "查找", "搜索", "查询","DrugBAN"
]


# 配置基础设置
Settings.embed_model = OllamaEmbedding(
    model_name="deepseek-r1:7b"
)
Settings.llm = Ollama(
    model="deepseek-r1:7b",
    request_timeout=360
)

def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def send_email(sender_email, sender_authorization_code, recipient_email, subject, body):
    # 创建 MIMEMultipart 对象
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject

    message.attach(MIMEText(body, "plain"))

    # 创建 SMTP_SSL 会话
    with smtplib.SMTP_SSL("smtp.163.com", 465) as server:
        server.login(sender_email, sender_authorization_code)
        text = message.as_string()
        server.sendmail(sender_email, recipient_email, text)

# 提示词
def should_query_knowledge_base(prompt: str) -> bool:
    """
    判断是否应该启动知识库查询
    """
    # 使用正则表达式匹配关键词
    pattern = r"(知识库|文档|资料|信息|notion|检索|查找|搜索|查询)"
    if re.search(pattern, prompt, re.IGNORECASE):
        return True
    
    # 使用模糊匹配检查输入是否包含 Prompt 库中的关键词
    for keyword in KNOWLEDGE_BASE_KEYWORDS:
        match_score = process.extractOne(keyword, [prompt])[1]
        if match_score >= 80:  # 相似度阈值
            return True
    return False

# 知识库查询-无向量数据库
def query_knowledge_base(question: str):
    
    # """
    # 查询个人知识库并返回答案
    # """
    # 配置路径
    md_folder = "./data/mds"  # 修改为你的Markdown文件夹路径
    
    # 加载并处理文档
    documents = SimpleDirectoryReader(md_folder).load_data()
    index = VectorStoreIndex.from_documents(documents)
    # 缓存索引对象避免重复加载
    # if "knowledge_index" not in st.session_state:
    #     st.session_state.knowledge_index = VectorStoreIndex.from_documents(documents)
    # index = st.session_state.knowledge_index
    # 创建查询引擎
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        verbose=True  # 显示检索过程
    )
    # 添加回答模板
    prompt_template = f"""
    [指令] 请用以下格式回答：
    <thinking>你的思考过程（分析用户意图，检索策略等）</thinking>
    <answer>结构化回答内容</answer>
    """
    # 执行查询
    response = query_engine.query(prompt_template + "\n\n用户问题：" + question)
    return str(response)




# 新增响应解析函数
def parse_response(raw_response: str) -> dict:
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', raw_response, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', raw_response, re.DOTALL)
    
    return {
        "thinking": thinking_match.group(1).strip() if thinking_match else "",
        "answer": answer_match.group(1).strip() if answer_match else raw_response
    }


def main():
    
    
    ###########---ui----############
    st.title("🪪 Knowing")

    # Initialize chat history
    # 初始化聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 在侧边栏添加控制项
    if st.sidebar.button("清空历史"):
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # 在侧边栏添加模型选择
#     selected_model = st.sidebar.selectbox(
#     "选择模型",
#     ("deepseek-r1:7b", "gpt-4")
# )
    # 性能监控
    start = time.time()
    # 上传文档
    # with st.sidebar:
    #     st.header("📁 知识库管理")  # 新增侧边栏模块
    #     uploaded_files = st.file_uploader(
    #         "上传知识文档（支持PDF/Word/Markdown/TXT）",
    #         type=["pdf", "docx", "md", "txt"],
    #         accept_multiple_files=True,
    #         help="每次上传后自动更新知识库"
    #     )
    ###########---ui----############
    # upload_files(uploaded_files)
    

    # st.chat_input：创建输入框，提示用户输入消息
    # 若用户输入，则将其赋值给promt变量
    if prompt := st.chat_input("What is your message?"):
        # Display user message in chat message container
        # 使用 st.chat_message 显示用户输入的消息，角色为 "user"
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 判断是否应该启动知识库查询
        if should_query_knowledge_base(prompt):
            print("启动知识库查询")
            # 在main()中替换原有响应处理逻辑
            # knowledge_response = query_knowledge_base(prompt)
            knowledge_response = query_with_RAG(prompt)
            
            parsed = parse_response(knowledge_response)  # 新增解析步骤
            print("知识库查询完成")
            
            # 构建结构化消息内容
            formatted_response = f"""
            ​**思考过程**:\n{parsed['thinking']}\n\n
            ​**正式回答**:\n{parsed['answer']}
            """
            with st.chat_message("assistant"):
                st.markdown(f"**思考过程**:\n{parsed['thinking']}")  # 可选显示思考过程
                st.markdown(f"**正式回答**:\n{parsed['answer']}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": formatted_response
                })
            # # 显示并存储完整消息
            
        else:
            # Handle other types of prompts (e.g., chat completion)
            response = chat_completion_request(
                messages=st.session_state.messages,
                tools=tools
            )
            # st.sidebar.json(st.session_state)
            # st.sidebar.write(response)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                if content := response.choices[0].message.content:
                    st.markdown(content)
                    st.session_state.messages.append({"role": "assistant", "content": content})
                # RAG
                else:
                    fn_name = response.choices[0].message.tool_calls[0].function.name
                    fn_args = response.choices[0].message.tool_calls[0].function.arguments

                    def confirm_send_fn():
                        send_email(
                            sender_email=args["FromEmail"],
                            sender_authorization_code=AUTHORIZATION_CODE,
                            recipient_email=args["Recipients"],
                            subject=args["Subject"],
                            body=args["Body"],
                        )
                        st.success("邮件已发送")
                        st.session_state.messages.append({"role": "assistant", "content": "邮件已发送，还需要什么帮助吗？"})
                        # Refresh sidebar
                        st.sidebar.json(st.session_state)
                        st.sidebar.write(response)

                    def cancel_send_fn():
                        st.warning("邮件发送已取消")
                        st.session_state.messages.append({"role": "assistant", "content": "邮件已取消，还需要什么帮助吗？"})
                        # Refresh sidebar
                        st.sidebar.json(st.session_state)
                        st.sidebar.write(response)

                    if fn_name == "send_email":
                        args = json.loads(fn_args)
                        st.markdown("邮件内容如下：")
                        st.markdown(f"发件人: {args['FromEmail']}")
                        st.markdown(f"收件人: {args['Recipients']}")
                        st.markdown(f"主题: {args['Subject']}")
                        st.markdown(f"内容: {args['Body']}")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.button(
                                label="✅确认发送邮件",
                                on_click=confirm_send_fn
                            )
                        with col2:
                            st.button(
                                label="❌取消发送邮件",
                                on_click=cancel_send_fn
                            )
        
    st.sidebar.metric("响应时间", f"{time.time()-start:.2f}s")
    
if __name__ == "__main__":
    main()
   
