import torch
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
def explain_xlmr_ig_embeddings(text, model=None, tokenizer=None, target_class=1):
    if model is None:
        raise ValueError("Model argument is required!")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    device = next(model.parameters()).device

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device).bool()  # ensure bool dtype

    embedding_layer = model.roberta.embeddings.word_embeddings
    input_embeds = embedding_layer(input_ids)

    def forward_func(embeds):
        outputs = model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs.logits

    baseline_embeds = torch.zeros_like(input_embeds).to(device)
    ig = IntegratedGradients(forward_func)

    attributions, delta = ig.attribute(
        inputs=input_embeds,
        baselines=baseline_embeds,
        target=target_class,
        return_convergence_delta=True,
        n_steps=50,
    )

    attributions_sum = attributions.sum(dim=-1).squeeze(0).detach().cpu()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    valid_indices = [i for i, t in enumerate(tokens) if t not in tokenizer.all_special_tokens]
    tokens_filtered = [tokens[i] for i in valid_indices]
    attr_filtered = attributions_sum[valid_indices]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(range(len(tokens_filtered)), attr_filtered, color="#9467bd")
    ax.set_xticks(range(len(tokens_filtered)))
    ax.set_xticklabels(tokens_filtered, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel("Attribution Score")
    ax.set_title("Integrated Gradients Attribution (Embedding Inputs)", pad=20)

    threshold = 0.5 * attr_filtered.max()
    for bar, score in zip(bars, attr_filtered):
        if score > threshold:
            bar.set_color("#d62728")

    plt.tight_layout()
    return fig


# Usage example:
# from transformers import AutoModelForSequenceClassification
# model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base-finetuned-toxic")
# model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
# fig = explain_xlmr_ig_embeddings("You are an idiot!", model)
# plt.show()
