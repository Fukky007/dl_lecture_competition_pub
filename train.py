import torch
import hydra
from omegaconf import DictConfig

import pandas
from src.utils import set_seed
from tokenizers import BertWordPieceTokenizer

@hydra.main(version_base=None, config_path="configs", config_name="base")
def train(args: DictConfig):
    set_seed(args.seed)

    # ------------------
    #    Dataloader
    # ------------------
    df = pandas.read_json("./data/train.json")  # 画像ファイルのパス，question, answerを持つDataFrame
    text = []
    for question in df["question"]:
        text.append(question)

    # ------------------
    #       Model
    # ------------------

    # ------------------
    #  Loss & optimizer
    # ------------------

    # ------------------
    #   Start training
    # ------------------

    # Initialize an empty tokenizer
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=True,
        lowercase=True,
    )

    # And then train
    tokenizer.train_from_iterator(
        text,
        vocab_size=10000,
        min_frequency=2,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        limit_alphabet=1000,
        wordpieces_prefix="##",
    )

    # Save the files
    tokenizer.save_model("./pre_train", "bert_wordpiece")

if __name__ == "__main__":
    train()
