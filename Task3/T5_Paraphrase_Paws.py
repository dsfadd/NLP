import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to('cuda')

def paraphrase(
        question,
        num_return_sequences=5,
        temperature=0.7,
):
    text = "paraphrase: " + question + " </s>"
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")

    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        do_sample=True,
        top_k=186,
        top_p=0.95,
        temperature=temperature,
        early_stopping=True,
        num_return_sequences=num_return_sequences
    )
    res = tokenizer.batch_decode(outputs, skip_special_tokens=True,clean_up_tokenization_spaces=True)

    return res