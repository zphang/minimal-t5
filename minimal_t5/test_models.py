import transformers
import minimal_t5.minimal_t5 as minimal_t5
import torch


def main():
    torch.set_grad_enabled(False)
    tokenizer = transformers.T5TokenizerFast.from_pretrained("t5-base")

    # Make Batch
    texts = ["Sentence 1.", "This is the second example."]
    labels = ["Sentence 2.", "This is more of the second example."]
    batch = tokenizer(texts, padding=True, return_tensors="pt")
    label_ids = tokenizer(labels, padding=True, return_tensors="pt")["input_ids"]
    label_ids[label_ids == 0] = -100
    batch["labels"] = label_ids

    # Create HF Model
    hf_model = transformers.T5ForConditionalGeneration.from_pretrained("google/t5-base-lm-adapt")
    hf_model = hf_model.cuda().eval()

    # Create T5 Simple Model
    config = minimal_t5.T5BaseConfig
    simple_model = minimal_t5.T5Model(config=config)
    simple_model.load_state_dict(minimal_t5.get_hf_state_dict("google/t5-base-lm-adapt"))
    simple_model = simple_model.cuda().eval()

    # 1. Test forward-pass logits
    hf_out = hf_model(
        input_ids=batch["input_ids"].cuda(),
        attention_mask=batch["attention_mask"].cuda(),
        labels=batch["labels"].cuda()
    )
    simple_logits = simple_model(
        encoder_input_ids=batch["input_ids"].cuda(),
        decoder_input_ids=minimal_t5.shift_right(batch["labels"].cuda()),
    )
    assert (simple_logits == hf_out.logits).all()
    print("[Logits] Passed!")

    # 1b. Test loss
    simple_loss = minimal_t5.compute_logits_loss(lm_logits=simple_logits, labels=batch["labels"].cuda())
    assert simple_loss == hf_out.loss
    print("[Loss] Passed!")

    # 2. Test Generation
    hf_out = hf_model.generate(
        input_ids=batch["input_ids"].cuda(),
        attention_mask=batch["attention_mask"].cuda(),
        max_length=20,
    )
    simple_out = simple_model.generate(
        encoder_input_ids=batch["input_ids"].cuda(),
        generation_length=19,
    )
    assert (simple_out == hf_out).all()
    print("[Generation] Passed!")


if __name__ == "__main__":
    main()
