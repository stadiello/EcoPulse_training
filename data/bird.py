from datasets import load_dataset

dataset = load_dataset(
    "greenarcade/wav2vec2-vd-bird-sound-classification-dataset"
)

print(dataset)
print(dataset["train"][0])