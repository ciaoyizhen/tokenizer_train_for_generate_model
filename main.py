import typer
from typing import Annotated


app = typer.Typer()

@app.command("train")
def train(
    *,
    data: Annotated[list[str], typer.Option(
            help="训练tokenizer的数据"
        )]=["data/斗破苍穹.txt", "data/武动乾坤.txt"],
    vocab_size: Annotated[int, typer.Argument(
        help="训练的词典大小"
    )]=25000,
    save_dir: Annotated[str, typer.Argument(
        help="tokenizer保存位置"
    )]="output/tctd_model"
):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(data,
                    vocab_size=vocab_size,
                    special_tokens=[
                        "<|startoftext|>",
                        "<|endoftext|>",
                        "<|user|>",
                        "<|system|>",
                        "<|assistant|>"
                        ]  # 这里虽然说 没有计划做对话系统，但是先放进去
                    )

    my_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = tokenizer,
        bos_token="<|startoftext>",
        eos_token="<|endoftext|>"
    )
    my_tokenizer.save_pretrained(save_dir)


@app.command("infer")
def infer(
    *,
    test_string: Annotated[str, typer.Argument(
        ...,
        help="测试分词的文字"
    )],
    model_path: Annotated[str, typer.Argument(
        ...,
        help="模型位置"
    )]="output/tctd_model"
):
    from rich import print
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"[yellow bold] 是否是快速分词器: {tokenizer.is_fast}")
    print(f"[green bold] 你的输入是:{test_string}")
    model_input = tokenizer(test_string)
    print(f"[red bold] 模型tokenizer后的结果:{model_input}")
    input_ids = model_input["input_ids"]
    print(f"[blue bold]模型反转回来的结果:{tokenizer.decode(input_ids)}")


    
if __name__ == "__main__":
    app()