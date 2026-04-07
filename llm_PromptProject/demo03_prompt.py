# 1 导入必备的工具包
from dotenv import load_dotenv
import openai
import os

# 加载环境变量
load_dotenv()

# 提供相似，不相似的语义匹配例子
examples = {
    '是': [
        ('公司ABC发布了季度财报，显示盈利增长。', '财报披露，公司ABC利润上升。'),
    ],
    '不是': [
        ('黄金价格下跌，投资者抛售。', '外汇市场交易额创下新高。'),
        ('央行降息，刺激经济增长。', '新能源技术的创新。')
    ]
}

sentence_pairs = [
    ('股票市场今日大涨，投资者乐观。', '持续上涨的市场让投资者感到满意。'),
    ('油价大幅下跌，能源公司面临挑战。', '未来智能城市的建设趋势愈发明显。'),
    ('利率上升，影响房地产市场。', '高利率对房地产有一定冲击。'),
]


# 2 构建函数，进行prompt设计（描述清楚任务及输出格式）
def init_prompts(examples):
    """
    初始化前置prompt，便于模型做 incontext learning。
    """
    pre_history = [{"role": "system",
                    "content": '现在你需要帮助我完成文本匹配任务，当我给你两个句子时，你需要回答我这两句话语义是否相似。只需要回答是否相似，不要做多余的回答。'}
                   ]

    for key, sentence_pairs in examples.items():
        for sentence_pair in sentence_pairs:
            sentence1, sentence2 = sentence_pair
            pre_history.append({
                "role": 'user',
                "content": f'句子一: {sentence1}\n句子二: {sentence2}\n上面两句话是相似的语义吗？'
            })
            pre_history.append({
                "role": 'assistant',
                "content": key
            })

    return {'pre_history': pre_history}


# 3 构建推理函数
def model_chat(content: str, history=[]) -> str:
    """
    调用大模型对话接口
    :param messages: 输入内容
    :param model: 模型名称
    :return: 大模型输出内容 str
    """
    client = openai.OpenAI(
        base_url=os.getenv("base_url"),
        api_key=os.getenv("api_key"),
    )
    messages = [{"role": "user", "content": content}]
    response = client.chat.completions.create(
        model=os.getenv("model"),
        messages=history + messages,
        stream=False,
    )
    text = response.choices[0].message.content
    return text


# 4 构建后处理函数

# 5 调用推理+后处理
prompts_info = init_prompts(examples)
for sentence_pair in sentence_pairs:
    sentence1, sentence2 = sentence_pair
    sentence_with_prompt = f'句子一: {sentence1}\n句子二: {sentence2}\n上面两句话是相似的语义吗？'
    result=model_chat(sentence_with_prompt, history=prompts_info['pre_history'])
    print(f'sentence_pair-->{sentence_pair}')
    print(f'result-->{result}')