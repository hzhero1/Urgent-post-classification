import joblib
from openai import OpenAI
import joblib
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from textblob import TextBlob

# 需要与训练时相同的 text_features 定义，供 joblib 反序列化时查找
# 如果没有这个函数，加载带 FunctionTransformer(pickle) 的 pipeline 会失败

def text_features(texts):
    features = []
    for t in texts:
        if not isinstance(t, str):
            t = ""
        tokens = word_tokenize(t)
        length = len(tokens)
        avg_len = np.mean([len(w) for w in tokens]) if tokens else 0
        pos_counts = pos_tag(tokens)
        nouns = sum(1 for w, p in pos_counts if p.startswith("NN"))
        verbs = sum(1 for w, p in pos_counts if p.startswith("VB"))
        sentiment = TextBlob(t).sentiment.polarity
        features.append([length, avg_len, nouns, verbs, sentiment])
    return np.array(features)

prompt = """You are a classifier that predicts the urgency level of posts in a MOOC discussion forum.

Input: a single post as a string of text.  
Output: a single integer from 1 to 7. Output only the integer.

Urgency levels:
1: No reason to read the post  
2: Not actionable, read if time  
3: Not actionable, may be interesting  
4: Neutral, respond if spare time  
5: Somewhat urgent, good idea to reply, a teaching assistant might suffice  
6: Very urgent: good idea for the instructor to reply  
7: Extremely urgent: instructor definitely needs to reply

Choose the level that best matches the urgency implied by the post."""

text = "I want to introduce yo you my latest project!"

client = OpenAI(
        api_key="sk-3981b366212a4d5da0257f5a63055416",
        base_url="https://api.deepseek.com")

# -------------------------------
# 8. 加载模型 & 新样本预测方法
# -------------------------------
loaded_pipeline = joblib.load("LR_classifier.pkl")

def classify_text(model, text):
    """
    输入: 已训练好的 pipeline, 文本字符串
    输出: 分类结果 (0 或 1)
    """
    return model.predict([text])[0]

while True:
    text = input("\n请输入待分类文本：")
    print("输入文本:", text)

    print("\n逻辑回归预测中...")
    lr_prediction = classify_text(loaded_pipeline, text)
    print("逻辑回归预测结果（0/1）:", lr_prediction)
    if lr_prediction == 1:
        print("逻辑回归预测该帖子为: 紧急\n")
    else:
        print("逻辑回归预测该帖子为: 不紧急\n")


    print("DeepSeek预测中...")
    # sk-3981b366212a4d5da0257f5a63055416
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"please classify the urgency level of the following post: {text}"},
        ],
        stream=False
    )

    ds_prediction = response.choices[0].message.content.strip()
    ds_label = 0
    print("DeepSeek预测结果（7级）:", ds_prediction)
    if ds_prediction in ["1","2","3"]:
        ds_label = 0
        print("DeepSeek预测该帖子为: 不紧急\n")
    elif ds_prediction in ["4","5","6","7"]:
        ds_label = 1
        print("DeepSeek预测该帖子为:紧急\n")

    print("综合判断:")
    if lr_prediction == 1 and ds_label:
        print("紧急\n")
    elif lr_prediction == 0 and ds_label == 0:
        print("不紧急\n")
    else:
        print("紧急可能\n")