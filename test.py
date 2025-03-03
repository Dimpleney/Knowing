# test 模型 问答性能
# import os
# from dotenv import load_dotenv

# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# import json
# from openai import OpenAI
# import streamlit as st
# from llama_index.core import VectorStoreIndex, Settings,SimpleDirectoryReader
# from llama_index.llms.ollama import Ollama
# from llama_index.embeddings.ollama import OllamaEmbedding

# # 配置基础设置
# Settings.embed_model = OllamaEmbedding(
#     model_name="deepseek-r1:7b"
# )
# Settings.llm = Ollama(
#     model="deepseek-r1:7b",
#     request_timeout=360
# )

# # 知识库查询
# def query_knowledge_base(question: str):
#     """
#     查询个人知识库并返回答案
#     """
#     # 配置路径
#     md_folder = "/root/code/llama-index/mds"  # 修改为你的Markdown文件夹路径
    
#     # 加载并处理文档
#     documents = SimpleDirectoryReader(md_folder).load_data()
#     index = VectorStoreIndex.from_documents(documents)

#     # 创建查询引擎
#     query_engine = index.as_query_engine(
#         similarity_top_k=3,
#         verbose=True  # 显示检索过程
#     )
    
#     # 执行查询
#     response = query_engine.query(question)
#     return str(response)


# def main():
#     # # 执行查询
#     # questions = [
#     #     "DrugBAN的创新点是什么？",
#     #     "DeepConv-DTI如何处理蛋白质序列？",
#     #     "介绍你的模型名称"
#     # ]
#     # for q in questions:
#     #     print(f"\n问题：{q}")
#     #     response = query_knowledge_base(q)
#     #     print(f"答案：{response}\n{'-'*50}")
    
#     response = query_knowledge_base("DrugBAN的创新点是什么")
#     print(response)

# if __name__ == "__main__":
#     main()


# # test
# KNOWLEDGE_BASE_KEYWORDS = [
#     "知识库", "知识库查询", "文档", "资料", "信息",
#     "notion", "Notion", "检索", "查找", "搜索", "查询"
# ]

# def should_query_knowledge_base(prompt: str) -> bool:
#     print("prompt:",prompt)
#     """
#     判断是否应该启动知识库查询
#     """
#     # 检查输入是否包含 Prompt 库中的关键词
#     return any(keyword in prompt for keyword in KNOWLEDGE_BASE_KEYWORDS)

# # 测试用例
# test_cases = [
#     "帮我从知识库中查找一下项目文档。",
#     "Notion 里有没有关于 AI 的资料？",
#     "检索一下最新的技术文档。",
#     "我想查一下资料",
#     "帮我搜索一下信息",
#     "今天天气怎么样？",  # 不应该触发
#     "给我讲个笑话"  # 不应该触发
# ]

# for test in test_cases:
#     print(f"Input: {test} -> Should query: {should_query_knowledge_base(test)}")

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext,Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# 双重配置：同时设置嵌入模型和LLM
Settings.embed_model = OllamaEmbedding(model_name="deepseek-r1:7b")
Settings.llm = Ollama(model="deepseek-r1:7b", request_timeout=360)

# 加载一些文档
documents = SimpleDirectoryReader("./data/mds").load_data()

# 初始化客户端，设置保存数据的路径
db = chromadb.PersistentClient(path="./chroma_db")

# 创建集合
chroma_collection = db.get_or_create_collection("quickstart")

# 将Chroma作为向量存储分配给上下文
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 创建索引
# 在创建索引时显式指定嵌入模型
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context
)

# 创建查询引擎并查询
# query_engine = index.as_query_engine()
# 创建查询引擎时显式指定LLM
query_engine = index.as_query_engine(
    llm=Settings.llm  # 确保使用本地模型
)
response = query_engine.query("What is the meaning of life?")
print(response)  # 输出查询结果
