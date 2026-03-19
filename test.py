from nltk.tokenize import word_tokenize

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

# 示例
sample_text = "Hello, world! This is urgent."
print(tokenize_text(sample_text))