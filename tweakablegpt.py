import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F


class GPTConfig:
    def __init__(self, vocab_size, max_position_embeddings, n_layer, n_head, n_embd):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd


class CustomAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, alpha=0.5):
        super(CustomAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Custom initialization for linear layers
        for name, param in self.qkv_proj.named_parameters():
            if "weight" in name:
                init.normal_(param, mean=0, std=alpha * (1 / embed_dim) ** 0.5)
        for name, param in self.out_proj.named_parameters():
            if "weight" in name:
                init.normal_(param, mean=0, std=alpha * (1 / embed_dim) ** 0.5)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim) # [B, L, nh, 3 * d]
        q, k, v = qkv.chunk(3, dim=-1)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v)) # [B nh L d]

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True, scale = 1/self.head_dim) # mup
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_length, self.embed_dim
        )
        output = self.out_proj(attn_output)

        return output


class GPTBlock(nn.Module):
    def __init__(self, config):
        super(GPTBlock, self).__init__()
        self.attention = CustomAttention(config.n_embd, config.n_head)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5)

        for name, param in self.mlp.named_parameters():
            if "weight" in name:
                init.normal_(param, mean=0, std=(1 / config.n_embd) ** 0.5)

    def forward(self, x):
        attn_output = self.attention(self.ln_1(x))
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config, alpha=0.5):
        super(GPTModel, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_position_embeddings, config.n_embd)
        )
        self.blocks = nn.Sequential(*[GPTBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

        init.normal_(self.head.weight, mean=0, std=alpha * (1 / config.n_embd))
        init.normal_(self.embed.weight, mean=0, std=alpha * 3.3)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        position_ids = torch.arange(
            0, input_ids.size(1), dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        hidden_states = []
        x = self.embed(input_ids) + self.pos_embed[:, : input_ids.size(1), :]
        if output_hidden_states:
            hidden_states.append(x)
        for block in self.blocks:
            x = block(x)
            if output_hidden_states:
                hidden_states.append(x)

        x = self.ln_f(x)
        logits = self.head(x).float()

        outputs = {"logits": logits}
        if input_ids is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            outputs["loss"] = loss

        if output_hidden_states:
            outputs["hidden_states"] = hidden_states

        return outputs


if __name__ == "__main__":
    import torch
    import torch.optim as optim

    def train_and_generate(model, sequence, config, device="cuda:0"):
        model.to(device)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        epochs = 500
        inputs = torch.tensor([sequence], dtype=torch.long).to(device)

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(inputs)
            loss = output["loss"]
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

        model.eval()
        input_ids = torch.tensor([[sequence[0]]], dtype=torch.long).to(device)
        generated_sequence = [sequence[0]]

        for _ in range(len(sequence) - 1):
            with torch.no_grad():
                output = model(input_ids)
                logits = output["logits"]

                predicted_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
                generated_sequence.append(predicted_token_id)

                input_ids = torch.cat(
                    [
                        input_ids,
                        torch.tensor([[predicted_token_id]], dtype=torch.long).to(
                            device
                        ),
                    ],
                    dim=1,
                )

        return generated_sequence

    config = GPTConfig(
        vocab_size=50257,
        max_position_embeddings=1024,
        n_layer=4,
        n_head=4,
        n_embd=768,
    )

    model = GPTModel(config)

    sequence = list(range(11))
    generated_sequence = train_and_generate(model, sequence, config)
    print("Generated Sequence:", generated_sequence)
