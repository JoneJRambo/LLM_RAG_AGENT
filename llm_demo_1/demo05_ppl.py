##############################单句话###############################
sentences = [['the', 'cat', 'sat']]
unigram = {'the': 0.5, 'cat': 0.4, 'sat': 0.2}

joint_prob = 1.0  # 初始化联合概率为 1.0
N = 0
# 遍历语料库中的每一个句子
for sentence in sentences:
    N += len(sentence)
    # 步骤 1: 计算整个句子的联合概率 P(W)
    # 对应公式: P(W) = P(w_1) * P(w_2) * ... * P(w_N)
    for word in sentence:
        prob = unigram[word]
        joint_prob *= prob  # 将每个词的概率连乘起来
    # 步骤 2: 计算该句子的困惑度 PPL(W)
    # 对应公式: PPL(W) = P(W)^(-1/N)
# 使用 a ** b 表示 a的b次方, 这行代码是公式的直接翻译
ppl = joint_prob ** (-1.0 / N)
# 计算最终在整个测试集上的平均困惑度
print("ppl困惑度-->", ppl)


##############################多句话###############################
# 定义语料库
sentences = [
['I', 'have', 'a', 'pen'],
['He', 'has', 'a', 'book'],
['She', 'has', 'a', 'cat']
]
# 定义语言模型
unigram = {
'I': 1/12,
'have': 1/12,
'a': 3/12,
'pen': 1/12,
'He': 1/12,
'has': 2/12,
'book': 1/12,
'She': 1/12,
'cat': 1/12
}
# 计算困惑度
sentence_prob = 1
length = 0
# 当有多个样本的时候，我们可以将其视为一个大的序列，通过遍历和连乘的方式，计算整个序列的概率
for sentence in sentences:
    length += len(sentence)
    for word in sentence:
        sentence_prob *= unigram[word]
# 根据公式计算困惑度
perplexity = pow(sentence_prob, -1/length)
print(f'困惑度-->{perplexity}')  # 8.12323