import torch

def mean_pooling(
    model_output: torch.Tensor,
        attention_mask: torch.Tensor) -> torch.Tensor:

    token_embeddings = model_output.last_hidden_state

    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )

    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def get_sentence_embedding(text: str,tokenizer,model) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        outputs = model.encoder(**inputs)

    embedding = mean_pooling(outputs, inputs["attention_mask"])

    return embedding.squeeze(0)