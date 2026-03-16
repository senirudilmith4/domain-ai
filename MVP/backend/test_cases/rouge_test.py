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
    "The learning outcomes for Programming Fundamentals (CM1601) are:Present competence in the algorithmic approach to problem solving (LO1), Apply fundamental programming concepts using a high-level programming language (LO2), Implement robust, maintainable programs that use object-orientated analysis and design principles (LO3), Apply alternative code constructs for a given use case study on an intelligent application (LO4), Students learn predictive analysis, regression modelling, and statistical techniques.",
    "The prerequisites for module CM1601: Programming Fundamentals are not explicitly stated in the provided university documents.",
    "The assessment method for module CM1604 is not explicitly stated in the provided university documents.",
    "The goal of learning module Data Structures and Algorithms for Artificial Intelligence is to provide the theory of algorithms and data structures, evaluate their performance using complexity analysis theory, and apply algorithms and data structures to real-world problems.",
    "The module CM1603 is Database Systems and Fundamentals, which covers underlying theories and principles of relational database management system (RDBMS), implementing a relational database system using SQL, and applying database design principles to real-world problems.",
    "No specific details are available as the recommended reading list is empty",
    "The prerequisite for Module CM2601: Object Oriented Development is CM1601 or equivalent.",
    "The module CM2601: Object Oriented Development is worth 15 credits."

]

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)

bleu_scores = []
rouge1_scores = []
rougeL_scores = []
f1_scores = []

import re

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

for ref, gen in zip(references, generated):

    ref = normalize(ref)
    gen = normalize(gen)

    bleu = sentence_bleu([ref.split()], gen.split())

    # BLEU
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