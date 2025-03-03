# 生成向量数据库
# import chromadb
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer, Settings
# from llama_index.llms.ollama import Ollama
# from llama_index.embeddings.ollama import OllamaEmbedding
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.core import StorageContext


# # 配置基础设置
# Settings.embed_model = OllamaEmbedding(
#     model_name="deepseek-r1:7b"
# )
# Settings.llm = Ollama(
#     model="deepseek-r1:7b",
#     request_timeout=360
# ) 

# # 读取文档
# documents = SimpleDirectoryReader("./data/mds").load_data()

# # 初始化 Chroma 客户端，指定数据存储路径为当前目录下的 chroma_db 文件夹
# db = chromadb.PersistentClient(path="./chroma_db")

# # 获取或创建名为 "quickstart" 的集合，如果该集合不存在，则创建它
# chroma_collection = db.get_or_create_collection("quickstart")

# # 使用上述集合创建一个 ChromaVectorStore 实例，以便 llama_index 可以与 Chroma 集合进行交互
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# # 创建一个存储上下文，指定向量存储为刚刚创建的 ChromaVectorStore 实例
# storage_context = StorageContext.from_defaults(vector_store=vector_store)


# # 构建索引
# index = VectorStoreIndex.from_documents(
#     documents, storage_context=storage_context, transformations=[SentenceSplitter(chunk_size=256)]
# )




# 查询
import hashlib
import os
import chromadb
from llama_cloud import SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# 设置嵌入模型和语言模型
# 配置基础设置
Settings.embed_model = OllamaEmbedding(
    model_name="deepseek-r1:7b"
)
Settings.llm = Ollama(
    model="deepseek-r1:7b",
    request_timeout=360
)

def query_with_RAG(question:str):
    
    # 配置基础设置
    Settings.embed_model = OllamaEmbedding(
        model_name="deepseek-r1:7b"
    )
    Settings.llm = Ollama(
        model="deepseek-r1:7b",
        request_timeout=360
    )
    
    # 初始化 Chroma 客户端，指定数据存储路径为当前目录下的 chroma_db 文件夹
    db = chromadb.PersistentClient(path="./chroma_db")

    # 获取或创建名为 "quickstart" 的集合，如果该集合不存在，则创建它
    chroma_collection = db.get_or_create_collection("quickstart")

    # 使用上述集合创建一个 ChromaVectorStore 实例，以便 llama_index 可以与 Chroma 集合进行交互
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 创建一个存储上下文，指定向量存储为刚刚创建的 ChromaVectorStore 实例
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 从存储的向量中加载索引
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

    # 配置检索器
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5,  # 返回最相似的前 n 个文档片段
    )

    # 配置响应合成器
    response_synthesizer = get_response_synthesizer()

    # 组装查询引擎
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,    
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

def update_vector_store():
    
    # 配置基础设置
    Settings.embed_model = OllamaEmbedding(
        model_name="deepseek-r1:7b"
    )
    Settings.llm = Ollama(
        model="deepseek-r1:7b",
        request_timeout=360
    )
    
    """增量更新 ChromaDB 向量数据库"""
    # 初始化 Chroma 客户端
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # 创建/获取集合（需与初始化设置一致）
    collection = chroma_client.get_or_create_collection(
        name="quickstart"
    )
    
    # 配置 Chroma 向量存储
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 加载所有文档（包括新旧）
    documents = SimpleDirectoryReader("./data/mds").load_data()
    
    # 创建增量索引
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True  # 显示进度条
    )
    
    # 持久化存储
    index.storage_context.persist(persist_dir="./chroma_db")


# 数据库插入
# collection.add(
#     documents=["doc1", "doc2", "doc3", ...],
#     embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
#     metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
#     ids=["id1", "id2", "id3", ...]
# )
# def add_docs(upload_file):
#     dir = "./data/mds"
#     file_path = os.path.join(dir,upload_file.name)
#     # step 1: 读取
#     document = SimpleDirectoryReader(file_path).load_data()
#     # 创建collection
#     db = chromadb.PersistentClient(path="./chroma_db")
#     collection = db.get_or_create_collection("quickstart")
#     # collection.add()
#     embeddings = []
#     metadatas = []
#     ids = []
#     collection.add(
#         documents=document,
#         embeddings=embeddings,
#         metadatas=metadatas,
#         ids=ids
#     )

def add_docs(upload_file):
    # 创建存储目录
    dir_path = "./data/mds"
    os.makedirs(dir_path, exist_ok=True)
    
    # 生成唯一文件名（防止覆盖）
    file_hash = hashlib.md5(upload_file.getvalue()).hexdigest()[:8]
    file_name = f"{file_hash}_{upload_file.name}"
    file_path = os.path.join(dir_path, file_name)
    
    # 保存文件到本地
    with open(file_path, "wb") as f:
        f.write(upload_file.getbuffer())
    
    # ====================
    # 1. 文档分块处理
    # ====================
    # 读取文档内容
    documents = SimpleDirectoryReader(
        input_files=[file_path],
        file_metadata=lambda _: {"source": file_name}  # 添加基础元数据
    ).load_data()
    
    # 中文优化分块器
    splitter = SentenceSplitter(
        chunk_size=256,          # 每个块约512 tokens
        chunk_overlap=64,        # 块间重叠64 tokens
        paragraph_separator="\n\n",
        secondary_chunking_regex=r"([。！？])"  # 按中文标点分句
    )
    
    # 执行分块
    nodes = splitter(documents)
    
    # ====================
    # 2. 元数据生成
    # ====================
    metadatas = []
    for idx, node in enumerate(nodes):
        metadata = node.metadata.copy()  # 继承文件元数据
        metadata.update({
            "chunk_id": idx,
            "text_hash": hashlib.md5(node.text.encode()).hexdigest()[:6],
            "char_length": len(node.text),
            "language": "zh"  # 假设是中文文档
        })
        metadatas.append(metadata)
    
    # ====================
    # 3. 嵌入向量生成
    # ====================
    # 初始化嵌入模型
    embed_model = OllamaEmbedding(
        model_name="deepseek-r1:7b",
        base_url="http://localhost:11434",
        embed_batch_size=32  # 批量生成提高效率
    )
    
    # 生成所有块的嵌入
    embeddings = [
        embed_model.get_text_embedding(node.text)
        for node in nodes
    ]
    
    # ====================
    # 4. 唯一ID生成
    # ====================
    ids = [
        f"{file_hash}_{idx:04d}"  # 格式示例：8c9f2a_0001
        for idx in range(len(nodes))
    ]
    
    # ====================
    # 5. 写入Chromadb
    # ====================
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="quickstart",
        metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
    )
    
    # 分批写入（避免内存溢出）
    batch_size = 100
    for i in range(0, len(nodes), batch_size):
        batch_nodes = nodes[i:i+batch_size]
        
        collection.add(
            documents=[node.text for node in batch_nodes],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

add_docs("夏天最暴利生意，直击年轻人软肋.txt")
q="夏天最暴利生意是什么？"
print(query_with_RAG(q))
    
    
