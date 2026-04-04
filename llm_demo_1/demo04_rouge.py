from rouge import Rouge


# 生成文本
generated_text ="it is a nice day today"

# 参考文本
reference_text = "today is a nice day"

# 计算 Rouge 指标
rouge = Rouge()
scores = rouge.get_scores(hyps=generated_text, refs=reference_text)
print("scores-->",scores)