import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载配置文件
load_dotenv()

# 创建大模型客户端
model = ChatOpenAI(model=os.getenv("model"),
                   api_key=os.getenv("api_key"),
                   base_url=os.getenv("base_url"),
                   temperature=0)

# 构建提示词
schema = {'金融': ['日期', '股票名称', '开盘价', '收盘价', '成交量']}

examples = {
    '金融': [
        {
            'content': '2023-01-10，股市震荡。股票古哥-D[EOOE]美股今日开盘价100美元，一度飙升至105美元，随后回落至98美元，最终以102美元收盘，成交量达到520000。',
            'answers': {
                '日期': ['2023-01-10'],
                '股票名称': ['古哥-D[EOOE]美股'],
                '开盘价': ['100美元'],
                '收盘价': ['102美元'],
                '成交量': ['520000'],
            }
        }
    ]
}

sentences = [
    '2023-02-15，寓意吉祥的节日，股票佰笃[BD]美股开盘价10美元，虽然经历了波动，但最终以13美元收盘，成交量微幅增加至460,000，投资者情绪较为平稳。',
    '2023-04-05，市场迎来轻松氛围，股票盘古(0021)开盘价23元，尽管经历了波动，但最终以26美元收盘，成交量缩小至310,000，投资者保持观望态度。',
]


def build_prompt():
    # 获取总的类别
    classes = schema.keys()
    # 构建提示词，初始化
    init_prompt = f"你一个信息抽取专家。参考{schema}，完成信息抽取任务。示例：\n"
    for class_name, example in examples.items():
        # for item in example:
        init_prompt += str(example)
        # print(init_prompt)
    return init_prompt


# 把提示词送给大模型
def call_lllm(prompt):
    # prompt: 上面构建好的提示词
    prompt += "请对如下文本进行信息抽取，只需要输出最终结果即可。文本如下：\n"
    # 遍历文本，送入大模型
    for sentence in sentences:
        result = model.invoke(prompt + sentence)
        # 打印结果
        print(f"输入：{sentence}\n输出：{result.content}\n")


if __name__ == '__main__':
    prompt = build_prompt()
    # print(prompt)
    call_lllm(prompt)
