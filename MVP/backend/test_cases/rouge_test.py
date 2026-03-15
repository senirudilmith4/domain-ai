import json
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score

# Load evaluation data
with open("D:\\OneDrive\\Documents\\IIT\\STAGE 02\\DSGP\\Domain AI\\MVP\\backend\\test_cases\\test.json") as f:
    data = json.load(f)

references = [item["reference"] for item in data]

# Example RAG answers (replace with your system outputs)
generated = [
    "CM1607 is a 3 credit module.",
    "Students learn predictive analysis, regression modelling, and statistical techniques."
]

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)

bleu_scores = []
rouge1_scores = []
rougeL_scores = []
f1_scores = []

for ref, gen in zip(references, generated):

    # BLEU
    bleu = sentence_bleu([ref.split()], gen.split())
    bleu_scores.append(bleu)

    # ROUGE
    rouge = scorer.score(ref, gen)
    rouge1_scores.append(rouge['rouge1'].fmeasure)
    rougeL_scores.append(rouge['rougeL'].fmeasure)

    # F1 (word overlap)
    ref_words = set(ref.split())
    gen_words = set(gen.split())

    common = ref_words.intersection(gen_words)
 
    precision = len(common) / len(gen_words)
    recall = len(common) / len(ref_words)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    f1_scores.append(f1)

print("Average BLEU:", sum(bleu_scores)/len(bleu_scores))
print("Average ROUGE-1:", sum(rouge1_scores)/len(rouge1_scores))
print("Average ROUGE-L:", sum(rougeL_scores)/len(rougeL_scores))
print("Average F1:", sum(f1_scores)/len(f1_scores))