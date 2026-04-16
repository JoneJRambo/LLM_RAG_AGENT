import json


class ChatSession:
    def __init__(self, system_prompt="你是一个乐于助人的 AI 助手。"):
        # 1. 初始化对话历史，system_prompt 定义了模型的角色设定
        self.history = [
            {"role": "system", "content": system_prompt}
        ]

    def add_message(self, role, content):
        """向历史记录添加一条消息"""
        self.history.append({"role": role, "content": content})

    def get_response(self, user_input):
        # 2. 将用户输入添加到历史中
        self.add_message("user", user_input)

        # 3. 调用大模型 API (此处为伪代码，请替换为你的 SDK 调用方式)
        # response = client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=self.history
        # )
        # assistant_reply = response.choices[0].message.content

        # 模拟响应结果
        assistant_reply = f"模拟回复：你刚才说的是 '{user_input}'"

        # 4. 将模型回复添加到历史中，保证下一轮上下文完整
        self.add_message("assistant", assistant_reply)

        return assistant_reply

    def clear_history(self):
        """清空历史（仅保留系统提示词）"""
        self.history = self.history[:1]


# --- 使用示例 ---
chat = ChatSession()

# 第一轮
print(chat.get_response("你好，我叫小明。"))
# 第二轮（模型已经记得你是小明）
print(chat.get_response("我刚才说的名字是什么？"))