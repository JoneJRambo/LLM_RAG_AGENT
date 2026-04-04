from modelscope.utils.import_utils import candidates
from nltk.translate.bleu_score import sentence_bleu
import warnings
warnings.filterwarnings("ignore")


candidates =["it","is","a","good","day","today"]
references =[["today","is","a","good","day"]]

result = sentence_bleu(references= references,hypothesis= candidates,weights=(1,))
print("result1-->", result)
result = sentence_bleu(references= references,hypothesis= candidates,weights=(0.5,0.5))
print("result2-->", result)
result = sentence_bleu(references= references,hypothesis= candidates,weights=(0.25,0.25,0.25,0.25))
print("result3-->", result)
