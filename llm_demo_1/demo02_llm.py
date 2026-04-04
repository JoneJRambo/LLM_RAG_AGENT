from transformers import AutoModelForCausalLM, AutoTokenizer


# 模型的名称/路径（需要设置成本地存放的路径）
model_name = r"G:\ITCAST\LLM\models\Qwen2.5-3B-Instruct"
# model_name = r"D:\models\Qwen2.5-0.5B"

# 加载模型
# torch_dtype="auto"：自动选择数据类型（如float16/float32），节省显存
# device_map="auto"：自动将模型映射到可用的GPU/CPU上
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "给我详细解释一下transformer架构，详细解释其四个组成部分，包括输入输出，编码器和解码器"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# 将对话转换成模型可识别的文本格式
# - tokenize=False 表示先不进行分词，只生成纯文本
# - add_generation_prompt=True 表示在末尾加上生成提示，引导模型继续生成
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 分词处理
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 调用模型进行文本生成
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512  # 最大生成长度
)
# 去除掉输入部分，只保留模型新生成的内容
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 解码生成的 token 序列，转换成可读文本
# - skip_special_tokens=True 表示跳过特殊符号（如<eos>、<pad>）
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f'response-->{response}')