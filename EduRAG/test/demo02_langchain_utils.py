"""
langchain支持3种模型：
* LLM
* ChatModel
* Embedding model
"""

####################LangChain组件：Models####################
# LLM
def dm01():
    # 0.x版本
    # from langchain.llms import Ollama
    # 1.x版本 方式一
    from langchain_community.llms import Ollama
    model = Ollama(model="deepseek-r1:1.5b", temperature=0)
    # 1.x版本 方式二
    # from langchain_ollama import OllamaLLM
    # model = OllamaLLM(model="deepseek-r1:1.5b", temperature=0)

    result = model.invoke("请给我讲个鬼故事")
    print("result-->", result)


# ChatModel
def dm02():
    from langchain_core.messages import HumanMessage, SystemMessage
    # 0.x版本
    # from langchain.chat_models import ChatOllama
    # 1.x版本
    # from langchain_community.chat_models import ChatOllama
    from langchain_ollama import ChatOllama
    model = ChatOllama(model="deepseek-r1:1.5b", temperature=0)
    messages = [
        SystemMessage(content="现在你是一个著名的歌手"),
        HumanMessage(content="给我写一首歌词")
    ]
    result = model.invoke(messages)
    print('result-->', result)
    print('result.content-->', result.content)


# embedding model
def dm03():
    # from langchain_community.embeddings import OllamaEmbeddings
    from langchain_ollama import OllamaEmbeddings
    model = OllamaEmbeddings(model="mxbai-embed-large", temperature=0)
    res1 = model.embed_query('这是第一个测试文档')
    print('res1-->', res1, len(res1))

    res2 = model.embed_documents(['这是第一个测试文档', '这是第二个测试文档'])
    print('res2-->', res2, len(res2))


####################LangChain组件：Prompt####################
# TODO 掌握：提示模板（非Chat版本）
def dm04():
    # from langchain import PromptTemplate # Langchain 0.x版本使用
    from langchain_core.prompts import PromptTemplate
    # from langchain_community.llms import Ollama  # Langchain 0.x版本使用
    from langchain_ollama import OllamaLLM

    # model = Ollama(model="qwen2.5:0.5b") # Langchain 0.x版本使用
    model = OllamaLLM(model="qwen2.5:0.5b")

    # 提示词模板
    prompt = PromptTemplate.from_template("我的邻居姓{lastname}， 他最近生了个儿子，请帮他儿子取个名字")
    prompt_str = prompt.format(lastname="王")
    print("result1-->", prompt_str)

    prompt = PromptTemplate(template="我的邻居姓{lastname}， 他最近生了个儿子，请帮他儿子取个名字", input_variables=['lastname'])
    prompt_str = prompt.format(lastname='王')
    print("result2-->", prompt_str)

    result = model.invoke(prompt_str)
    print("result-->", result)


# 了解：Langchain源码写法
def dm05():
    from langchain_core.prompts import ChatPromptTemplate
    template = ChatPromptTemplate(
        [
            ("system", "You are a helpful AI bot. Your name is {name}."),
            ("human", "Hello, how are you doing?"),
            ("ai", "I'm doing well, thanks!"),
            ("human", "{user_input}"),
        ])

    prompt_value = template.invoke(
        {
            "name": "Bob",
            "user_input": "What is your name?",
        })
    print("prompt_value-->", prompt_value)

# TODO 掌握：Chat版本提示词模板
def dm06():
    from langchain_community.chat_models import ChatOllama
    # 0.x版本
    # from langchain.prompts import ChatPromptTemplate
    # 1.x版本
    from langchain_core.prompts import ChatPromptTemplate

    # 创建原始模板
    template_str = """您是一位专业的鲜花店文案撰写员。\n
    对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？"""

    # 把字符串传给提示词模板
    prompt = ChatPromptTemplate.from_template(template=template_str)

    # 给模板中的变量赋值
    prompt_result = prompt.format_messages(price='68', flower_name='玫瑰')
    print("prompt_result-->", prompt_result)

    # 把提示词送给大模型
    model = ChatOllama(model="deepseek-r1:1.5b", temperature=0)
    result = model.invoke(prompt_result)
    print("result-->", result)


# 了解：FewShot提示词模板
def dm07():
    from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
    from langchain_ollama.llms import OllamaLLM
    model = OllamaLLM(model="qwen2.5:0.5b")

    examples = [
        {"word": "开心", "antonym": "难过"},
        {"word": "高", "antonym": "矮"},
    ]

    example_template = """
    单词: {word}
    反义词: {antonym}
    """

    example_prompt = PromptTemplate(
        input_variables=["word", "antonym"],
        template=example_template,
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="给出每个单词的反义词",
        suffix="单词: {input}\n反义词:",
        input_variables=["input"],
        example_separator="\n",
    )

    prompt_text = few_shot_prompt.format(input="粗")
    print(prompt_text)
    print('*' * 80)
    print(model.invoke(prompt_text))


####################LangChain组件：Chains####################
# TODO 掌握：Chains
def dm08():
    from langchain_core.prompts import PromptTemplate
    from langchain_community.llms import Ollama
    from langchain_classic.chains.llm import LLMChain

    # 定义模板
    template = "你是一个取名专家，我的邻居姓{lastname}，他生了个儿子，请给他儿子起个名字。"

    prompt = PromptTemplate(
        input_variables=["lastname"],
        template=template,
    )
    llm = Ollama(model="qwen2.5:0.5b")

    # 调用方式一：普通调用
    chain = LLMChain(llm=llm,
                     prompt=prompt)
    print('1-->', chain.invoke("王"))
    # 调用方式二：LCEL(Langchain 表达式语言)
    chain = prompt | llm
    # 执行链
    print('2-->', chain.invoke("王"))


# 了解：Chains 多条链
def dm09():
    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import OllamaLLM
    # 创建第一条链
    template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"

    first_prompt = PromptTemplate(
        input_variables=["lastname"],
        template=template, )

    # 实例化模型
    llm = OllamaLLM(model="deepseek-r1:1.5b")
    # LCEL表达式
    first_chain = first_prompt | llm

    # 创建第二条链
    second_prompt = PromptTemplate(
        input_variables=["child_name"],
        template="邻居的儿子名字叫{child_name}，给他起一个小名")

    second_chain = second_prompt | llm

    # 链接两条链
    overall_chain = first_chain | second_chain

    print(overall_chain)
    print('*' * 80)
    # 执行链，只需要传入第一个参数
    catchphrase = overall_chain.invoke("王")
    print(catchphrase)


####################OutputParser：对大模型输出结果进行解析（转换）####################
# 结果美化：字符串
def dm10():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import OllamaLLM

    # 实例化模型
    model = OllamaLLM(model="qwen2.5:0.5b")

    # 创建简单链
    prompt = ChatPromptTemplate.from_template("解释{topic}是什么？回答控制20字以内")
    chain1 = prompt | model
    result1 = chain1.invoke({"topic": "ai"})
    print(f"result1-->{result1}")

    # 添加字符串解析器
    parser = StrOutputParser()
    chain2 = prompt | model | parser
    result2 = chain2.invoke({"topic": "ai"})
    print(f"result2-->{result2}")


# 结果美化：列表
def dm11():
    from langchain_core.output_parsers import CommaSeparatedListOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import OllamaLLM

    # 实例化模型
    model = OllamaLLM(model="qwen2.5:0.5b")

    # 创建列表解析器
    parser = CommaSeparatedListOutputParser()

    # 创建带格式说明的提示模板
    # format_instructions = parser.get_format_instructions()
    # print(format_instructions)

    prompt = ChatPromptTemplate.from_template(
        "用中文列出{topic}的五个最重要特点。\n{format_instructions}"
    )

    # 组合组件
    chain = prompt | model | parser

    # 调用链
    result = chain.invoke({
        "topic": "大模型",
        "format_instructions": "你的回答应该是一个用逗号分隔的值的列表，例如：foo, bar, baz或 foo,bar,baz"
    })
    print(result)


# 结果美化：JSON
def dm12():
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import OllamaLLM

    # 实例化模型
    model = OllamaLLM(model="qwen2.5:0.5b")

    # 创建JSON解析器
    json_parser = JsonOutputParser()

    # 创建带格式说明的提示模板
    json_format_instructions = json_parser.get_format_instructions()
    print(json_format_instructions)

    json_prompt = ChatPromptTemplate.from_template(
        "生成一个包含{person}基本信息的JSON。应包括姓名、职业、年龄和技能列表, 不要包含任何注释或额外说明。\n{format_instructions}"
    )

    # 组合组件
    json_chain = json_prompt | model | json_parser

    # 调用链
    result = json_chain.invoke({
        "person": "雷军",
        "format_instructions": "返回一个JSON对象"
    })

    print(result)


# 结果美化：Pydantic类型校验
def dm13():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_ollama import OllamaLLM
    # pydantic
    from pydantic import BaseModel, Field
    from typing import List


    # 实例化模型
    model = OllamaLLM(model="qwen2.5:0.5b")

    # 定义Pydantic模型
    class Movie(BaseModel):
        title: str = Field(description="电影标题")
        director: str = Field(description="导演姓名")
        year: int = Field(description="上映年份")
        # todo : 单标签多分类 （每个电影只能有个类型）、 多标签多分类（一个电影可以有多个类型）
        genre: List[str] = Field(description="电影类型")
        rating: float = Field(description="评分（1-10）")

    # 创建Pydantic解析器
    pydantic_parser = PydanticOutputParser(pydantic_object=Movie)

    # 创建带格式说明的提示模板
    format_instructions = pydantic_parser.get_format_instructions()
    print('1-->', format_instructions)

    pydantic_prompt = ChatPromptTemplate.from_template(
        "生成一部{genre}电影的信息。\n{format_instructions}"
    )

    # 组合组件
    pydantic_chain = pydantic_prompt | model | pydantic_parser

    # 调用链
    movie_data = pydantic_chain.invoke({
        "genre": "科幻",
        "format_instructions": format_instructions
    })
    print('2-->', movie_data)


####################LangChain组件：Memory####################
def dm14():
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.messages import messages_to_dict, messages_from_dict
    import json

    # 1.创建一个ChatMessageHistory对象，用来存储对话信息
    history = ChatMessageHistory()
    # 添加用户消息
    history.add_user_message("在吗？")
    # 添加大模型消息
    history.add_ai_message("在")
    # 打印所有消息
    print(f'history.messages1-->{history.messages}')

    # 2.可以将history.messages中的信息保存到字典中，然后保存到数据库或者文件中，方便后续读取
    # 2.1 messages_to_dict()方法将history.messages中的信息转换成字典
    dicts = messages_to_dict(history.messages)
    print(f'dicts-->{dicts}')
    # 2.2 这里将dicts保存到文件中
    with open('history.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(dicts, indent=2, ensure_ascii=False))

    # 3.从文件中读取出字典，然后将字典转换成消息
    # 3.1 读取文件
    messages = json.load(open('history.json', 'r', encoding='utf-8'))
    # 3.2 然后将字典转换成消息
    chat_messages = messages_from_dict(messages)
    print(f'chat_messages2-->{chat_messages}')


####################LangChain组件：VectorStore####################
def dm15():
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import TextLoader
    from langchain_ollama import OllamaEmbeddings

    # 1.加载文档
    # 创建 TextLoader 对象
    loader = TextLoader('./pku.txt', encoding='utf-8')
    # 加载文档
    # metadata：描述数据的数据。数据大小，路径，创建日期，作者，权限...
    docs = loader.load()
    print(f'docs-->{docs}')

    # 2.将文档进行分块
    # 创建 CharacterTextSplitter 对象
    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=200, chunk_overlap=30)
    # 分块
    split_docs = text_splitter.split_documents(docs)
    print(f'split_docs-->{split_docs}')

    # 3.将分割后的文档存储到向量数据库中
    # 加载embedding模型
    embedding = OllamaEmbeddings(model="mxbai-embed-large")
    # 创建向量数据库，需要指定 存储的文档和向量模型名称以及持久化目录
    chromadaDB = Chroma.from_documents(documents=split_docs,
                                       embedding=embedding,
                                       persist_directory='./chroma_db')
    print("chromadaDB-->", chromadaDB)

    # 假如你的向量数据库已经存在，那么可以直接加载
    # chromadaDB = Chroma(persist_directory='./chroma_db', embedding_function=embedding)

    # 4.使用向量数据库进行查询
    query = "1937年北京大学发生了什么？"
    result = chromadaDB.similarity_search(query, k=1)
    print(f'result-->{result}')


####################LangChain组件：检索器####################
def dm16():
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader('./pku.txt', encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print("texts-->", texts)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    print("embeddings-->", embeddings)
    db = FAISS.from_documents(texts, embeddings)
    print("db-->", db)
    result = db.similarity_search("北京大学什么时候成立的", k=2)
    print("result-->", result)


def dm17():
    from langchain_community.document_loaders import UnstructuredFileLoader
    loader = UnstructuredFileLoader('衣服属性.txt', encoding='utf8')
    docs = loader.load()
    print("1-->", docs)
    print(len(docs))
    first_01 = docs[0].page_content[:4]
    print(first_01)
    print('*' * 80)
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader('衣服属性.txt', encoding='utf8')
    docs = loader.load()
    print("2-->", docs)
    print(len(docs))
    first_01 = docs[0].page_content[:6]
    print(first_01)


def demo18():
    from langchain_core.documents import Document
    from langchain_text_splitters import CharacterTextSplitter

    # 创建分词器 separator参数指的是分割的分隔符，chunk_size指的是分割出来的每个块的大小，chunk_overlap指的是每个块之间重叠的大小
    text_splitter = CharacterTextSplitter(separator=" ",
                                          chunk_size=5,    # TODO chunk_size指的是分割出来的每个块的大小
                                          chunk_overlap=1) # TODO chunk_overlap指的是每个块之间重叠的大小

    # 一句话分割
    result1 = text_splitter.split_text("a b c d e f")
    print(f'result1--->{result1}')

    # 多句话分割
    result2 = text_splitter.create_documents(["a b c d e f", "e f g h"])
    print(f'result2--->{result2}')

    # 多句话分割
    result3 = text_splitter.split_documents([Document(page_content="a b c d e f", metadata={"id": "1"})])
    print(f'result3--->{result3}')

def demo19():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # 4个参数：
    # 1. 块大小 2.重叠字符数 3. 长度衡量方式：传入一个函数进来，调用函数判断长度
    # 4. 分割符号： 按照列表中的分割符号，从左到右依次进行分割，直到满足块大小限制。 如果分割到最后一个符号未满足块大小，直接截断

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=30,  # 每个块最多 30 个字符
        chunk_overlap=6,  # 相邻分块会共享 6 个字符
        # 使用len，中文和英文没有区别；byte
        length_function=len,  # 用字符数来衡量长度
        separators=["\n\n", "\n", " ", ""]  # 会优先尝试按 \n\n 分段，如果太长，再按 \n → 空格 → 逐字符切分。
    )

    text = """
    人工智能正在快速发展，尤其是大语言模型的应用，正在改变人类的工作方式。
    它们可以帮助人们进行写作、代码生成、甚至是科研探索。
    相比之下，新能源的发展同样重要。
    电动车和太阳能正在逐渐替代传统能源，减少碳排放，对全球环境保护至关重要。
    """
    docs = text_splitter.split_text(text)
    print(docs)

if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    # dm04()
    # dm05()
    # dm06()
    # dm07()
    # dm08()
    # dm09()
    # dm10()
    # dm11()
    # dm12()
    # dm13()
    # dm14()
    # dm15()
    # dm16()
    # dm17()
    # demo18()
    demo19()