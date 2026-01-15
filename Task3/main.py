import chatgpt_paraphraser_on_T5_base
from chatgpt_paraphraser_on_T5_base import paraphrase
import torch.nn.functional as F

from cos_sim import get_sentence_embedding

tokenizer=chatgpt_paraphraser_on_T5_base.tokenizer
model=chatgpt_paraphraser_on_T5_base.model

original= "After your workout, remember to focus on maintaining a good water balance."
original_embedding=get_sentence_embedding(original,tokenizer,model)

paraphrases=paraphrase(original,10)

for paraphrase in paraphrases:
    paraphrase_embedding=get_sentence_embedding(paraphrase,tokenizer,model)
    c_s=F.cosine_similarity(original_embedding, paraphrase_embedding, dim=0).item()
    print(f"{paraphrase} {c_s}")

