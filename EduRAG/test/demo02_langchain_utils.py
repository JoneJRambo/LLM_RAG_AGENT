"""
langchain支持3种模型：
* LLM
* ChatModel
* Embedding model
"""


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


# 提示模板
def dm03():
    from langchain_community.chat_models import ChatOllama
    # 0.x版本
    # from langchain.prompts import ChatPromptTemplate
    # 1.x版本
    from langchain_core.prompts import ChatPromptTemplate

    # 创建原始模板
    template_str = """您是一位专业的鲜花店文案撰写员。\n
    对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
    # """

    # 根据原始模板创建LangChain提示模板
    promp_emplate = ChatPromptTemplate.from_template(template_str)
    prompt = promp_emplate.format_messages(price='50', flower_name=["玫瑰"], )
    print('prompt-->', prompt)
    # prompt--> [HumanMessage(content="您是一位专业的鲜花店文案撰写员。\n\n对于售价为 50 元的 ['玫瑰'] ，您能提供一个吸引人的简短描述吗？\n# ", additional_kwargs={}, response_metadata={})]

    # 实例化模型
    model = ChatOllama(model="deepseek-r1:1.5b", temperature=0)

    # 打印结果
    result = model.invoke(prompt)
    print(result.content)
    # 当然可以！"玫瑰，售价 50 元，是爱情与浪漫的象征。每一朵都是精心挑选和包装，确保其完美无瑕。无论是在庆祝特别的日子还是简单的日常问候中，这束玫瑰都能传达你的爱意。现在就为你的爱人或自己选择一份特别的礼物吧！"


# embedding model
def dm04():
    from langchain_community.embeddings import OllamaEmbeddings

    model = OllamaEmbeddings(model="mxbai-embed-large", temperature=0)
    res1 = model.embed_query('这是第一个测试文档')
    print(res1)

    res2 = model.embed_documents(['这是第一个测试文档', '这是第二个测试文档'])
    print(res2)


if __name__ == '__main__':
    # dm01()
    dm02()
    # dm03()
    # dm04()