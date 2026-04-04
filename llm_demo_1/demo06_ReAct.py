import dashscope
import re
import time
from datetime import datetime
from dotenv import load_dotenv
import os
# 加载 .env 文件
load_dotenv()

# 设置你的 DashScope API Key
dashscope.api_key = os.getenv("api_key")

# 工具1：获取当前日期（返回格式：YYYY年MM月DD日）
def get_current_date():
    """返回当前日期，格式：2025年9月15日"""
    now = datetime.now()
    return f"{now.year}年{now.month}月{now.day}日"


# 工具2：查询节假日（根据月份查询）
def search_holidays(month):
    """
    查询指定月份的法定节假日
    month: 月份字符串，如 "9月"
    """
    # 模拟节假日数据（实际应用中应调用真实API）
    holidays = {
        "1月": ["元旦：1月1日"],
        "2月": ["春节：1月28日-2月3日"],
        "3月": [],
        "4月": ["清明节：4月4日-6日"],
        "5月": ["劳动节：5月1日-5日", "端午节：5月31日-6月2日"],
        "6月": [],
        "7月": [],
        "8月": [],
        "9月": [],  # 2025年9月没有法定节假日
        "10月": ["中秋节：10月6日-8日", "国庆节：10月1日-7日"],
        "11月": [],
        "12月": ["元旦：12月31日"]  # 实际元旦在1月，这里仅作示例
    }

    # 获取指定月份的节假日
    holidays_list = holidays.get(month, [])

    if holidays_list:
        return f"2025年{month}有以下法定节假日：\n" + "\n".join(holidays_list)
    else:
        return f"2025年{month}没有法定节假日。"


# 工具注册
TOOLS = {
    "get_current_date": get_current_date,
    "search_holidays": search_holidays
}


# 调用Qwen模型
def call_qwen(prompt):
    response = dashscope.Generation.call(
        model='qwen-max',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.5
    )
    if response.status_code == 200:
        return response.output['text']
    else:
        return f"Error: {response.message}"


# ReAct主循环
def react_solve(question):
    print(f"问题：{question}\n")
    steps = []  # 用来存储每一步的输出
    max_iterations = 5
    print("开始ReAct推理流程...\n")

    for i in range(max_iterations):
        # 构建上下文（包含之前的所有步骤）
        context = "\n".join(steps)
        prompt = f"""
你是一个使用ReAct范式的智能代理，必须按以下格式输出：
Thought: <你的思考>
Action: <要执行的动作，从 [{', '.join(TOOLS.keys())}] 中选择，或 Final Answer>
Action Input: <动作输入>

当前上下文：
{context}

问题：{question}
"""

        # 调用Qwen生成下一步
        output = call_qwen(prompt)
        print(f"模型输出（第{i + 1}步）：\n{output}\n")

        # 解析输出
        thought_match = re.search(r"Thought:\s*(.*)", output)
        action_match = re.search(r"Action:\s*(.*)", output)
        input_match = re.search(r"Action Input:\s*(.*)", output)

        if not thought_match or not action_match:
            steps.append(f"Error: 无法解析输出格式。输出: {output}")
            continue

        thought = thought_match.group(1).strip()
        action = action_match.group(1).strip()
        action_input = input_match.group(1).strip() if input_match else ""

        # 记录步骤
        steps.append(f"Thought: {thought}")
        steps.append(f"Action: {action}")

        # 如果是最终答案
        if action == "Final Answer":
            print("✅ 任务完成！最终答案：")
            print(f"   {action_input}\n")
            return action_input

        # 执行工具
        if action in TOOLS:
            print(f"执行工具: {action} | 输入: {action_input}")
            try:
                # 传递参数给工具
                if action == "search_holidays":
                    # 从输入中提取月份（如"9月"）
                    month = re.search(r"(\d+)月", action_input)
                    if month:
                        action_input = month.group(1) + "月"
                    else:
                        action_input = "9月"  # 默认9月
                    result = TOOLS[action](action_input)
                else:
                    result = TOOLS[action]()

                steps.append(f"Action Input: {action_input}")
                steps.append(f"Observation: {result}")
                print(f"Observation: {result}\n")
                time.sleep(0.5)  # 避免频繁调用
            except Exception as e:
                result = f"工具执行错误: {str(e)}"
                steps.append(f"Action Input: {action_input}")
                steps.append(f"Observation: {result}")
                print(f"Observation: {result}\n")
        else:
            result = f"无效动作: {action}"
            steps.append(f"Action Input: {action_input}")
            steps.append(f"Observation: {result}")
            print(f"Observation: {result}\n")

    # 超出最大迭代次数
    final_answer = "无法在限定步数内完成任务。"
    print(f"❌ 任务失败: {final_answer}")
    return final_answer


# 🚀 运行示例
if __name__ == "__main__":
    question = "这个月有几个法定节假日？分别是什么？"
    result = react_solve(question)