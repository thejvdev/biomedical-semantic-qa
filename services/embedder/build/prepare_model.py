from pathlib import Path
import torch
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "bge-m3"


class BGEM3ONNXWrapper(torch.nn.Module):
    def __init__(self, model_name="BAAI/bge-m3"):
        super().__init__()
        bge_m3 = BGEM3FlagModel(model_name, use_fp16=False)
        self.encoder = bge_m3.model.model
        self.sparse_linear = bge_m3.model.sparse_linear
        self.colbert_linear = bge_m3.model.colbert_linear

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        last_hidden_state = outputs.last_hidden_state

        dense_vecs = last_hidden_state[:, 0]
        dense_vecs = torch.nn.functional.normalize(dense_vecs, p=2, dim=-1)

        sparse_vecs = self.sparse_linear(last_hidden_state)
        sparse_vecs = torch.relu(sparse_vecs)
        sparse_vecs = torch.log1p(sparse_vecs)
        mask = attention_mask.unsqueeze(-1).bool()
        sparse_vecs = torch.where(mask, sparse_vecs, torch.zeros_like(sparse_vecs))
        sparse_vecs = sparse_vecs.squeeze(-1)

        colbert_vecs = self.colbert_linear(last_hidden_state)
        colbert_vecs = torch.nn.functional.normalize(colbert_vecs, p=2, dim=-1)

        return dense_vecs, sparse_vecs, colbert_vecs


def main():
    wrapper = BGEM3ONNXWrapper().eval()

    dummy_input_ids = torch.randint(0, 250002, (1, 16))
    dummy_attention_mask = torch.ones((1, 16), dtype=torch.int64)

    MODEL_DIR.mkdir(exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
            MODEL_DIR / "model.onnx",
            input_names=["input_ids", "attention_mask"],
            output_names=["dense", "sparse", "colbert"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "dense": {0: "batch_size"},
                "sparse": {0: "batch_size", 1: "seq_len"},
                "colbert": {0: "batch_size", 1: "seq_len"},
            },
            opset_version=18,
        )

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    tokenizer.save_pretrained(str(MODEL_DIR))


if __name__ == "__main__":
    main()
