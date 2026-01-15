from transformers import T5ForConditionalGeneration, T5Tokenizer, GenerationConfig


def paraphrase(
        question,
        repetition_penalty=1.1,
        temperature=0.1,
        max_length=64
):
    inputs = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt",
        max_length=max_length,
        truncation=True
    ).to(device)

    outputs = model.generate(
    **inputs,
    max_length=max_length,
    num_beams=5,
    do_sample=False,
    temperature=temperature,
    repetition_penalty=repetition_penalty,
)

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

device="cuda"
model = T5ForConditionalGeneration.from_pretrained("./fine_tuned_paraphraser").to(device)
tokenizer = T5Tokenizer.from_pretrained("./fine_tuned_paraphraser")

sentence = "Практика повышает уверенность в навыках."

print(paraphrase(question=sentence))
