import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import numpy as np
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from textblob import TextBlob
from sklearn.metrics import accuracy_score, f1_score, classification_report

# -------------------------------
# 1. 自定义统计特征
# -------------------------------

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

# 清洗文本数据，去掉不可见字符和多余空格
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # 去掉 ASCII 控制字符
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # 去掉常见不可见 Unicode 符号
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    # 去掉 Excel 转义的不可见字符，例如 _x0007_
    text = re.sub(r'_x[0-9A-Fa-f]{4}_', '', text)
    # 去掉多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 生成新的数据集，包含清洗后的文本和二分类标签
def produce_full_dataset(path_data, clean_text):
    df = pd.read_excel(path_data)
    df = df[["Text", "Urgency(1-7)"]]
    df["Text"] = df["Text"].apply(clean_text)
    df.rename(columns={"Urgency(1-7)": "Label"}, inplace=True)
    df["Label"] = (df["Label"] >= 4).astype(int)
    df.to_csv("data_processed.xlsx", index=False, encoding="utf-8-sig")
    print("数据清洗并保存成功！")


def produce_testset(path_data):
        # 1. 读取数据
    df = pd.read_csv(path_data)

    # 2. 分离特征和目标（只是为了划分，不丢失数据）
    X = df.drop('Label', axis=1)  # 这里X包含除了target外的所有列
    y = df['Label']                # 目标列

    # 3. 划分数据集（只划分索引，不丢失任何列）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    # 4. 根据索引从原DataFrame提取完整的行
    train_df = df.loc[X_train.index]  # 保留所有列
    test_df = df.loc[X_test.index]    # 保留所有列

    # 5. 导出测试集
    test_df.to_csv('testset.csv', index=False)

    print(f"测试集已导出，共 {len(test_df)} 条数据")
    print(f"测试集包含的列：{list(test_df.columns)}")
    print(f"测试集前3行：")
    print(test_df.head(3))


def random_select_csv(input_file, output_file, n, random_state=42):
    # 读取数据
    df = pd.read_csv(input_file)
    
    # 检查n是否大于总数据量
    if n > len(df):
        print(f"警告：请求的{n}条大于总数据量{len(df)}条，将返回全部数据")
        n = len(df)
    
    # 随机选择n条数据
    sampled_df = df.sample(n=n, random_state=random_state)
    
    # 导出
    sampled_df.to_csv(output_file, index=False)
    
    print(f"已从{input_file}中随机选择{n}条数据")
    print(f"导出到：{output_file}")
    print(f"数据预览：")
    print(sampled_df.head())
    
    return sampled_df



def eval_model_on_csv(model, csv_path, text_col="Text", label_col="Label"):
    df = pd.read_csv(csv_path)
    X = df[text_col].astype(str)
    y = df[label_col].astype(int)

    y_pred = model.predict(X.tolist())

    print("accuracy:", accuracy_score(y, y_pred))
    print("f1:", f1_score(y, y_pred, average="binary"))
    print(classification_report(y, y_pred))

def eval_AI_label_csv(csv_path, pred_col="Prediction", label_col="Label"):
    """
    针对已经有模型预测列（pred）和真实标签列（label）的 csv 评估。
    返回 accuracy/f1，并打印 classification_report。
    """
    df = pd.read_csv(csv_path)

    if pred_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{pred_col}' and '{label_col}'")

    y_pred = df[pred_col].astype(int)
    y_true = df[label_col].astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")  # 二分类；多分类可改 macro/weighted
    print(f"accuracy = {acc:.4f}")
    print(f"f1 = {f1:.4f}")
    print(classification_report(y_true, y_pred))
    return {"accuracy": acc, "f1": f1}

# df = pd.read_csv("sampled_100_copilot_fewshot_labeled.csv")
# df = df[["Text", "Label"]]
# df["Label"] = (df["Label"] >= 4).astype(int)
# df.to_csv("sampled_100_copilot_fewshot_labeled_binary.csv", index=False, encoding="utf-8-sig")
# print("数据清洗并保存成功！")

print("AI evaluation:")
eval_AI_label_csv("sampled_100_copilot_labeled_binary.csv", pred_col="Prediction", label_col="Label")

print("AI evaluation-fewshot:")
eval_AI_label_csv("sampled_100_copilot_fewshot_labeled_binary.csv", pred_col="Prediction", label_col="Label")

print("AI evaluation binary:")
eval_AI_label_csv("sampled_100_copilot2.csv", pred_col="Prediction", label_col="Label")

print("LR evaluation:")
LR_classifier = joblib.load("LR_classifier.pkl")
print(LR_classifier.predict(["This is a test sentence."]))
eval_model_on_csv(LR_classifier, "sampled_100.csv")

# Prompt 1 for AI
"""

You are a classifier that predicts the urgency level of posts in a MOOC discussion forum.

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

Choose the level that best matches the urgency implied by the post.

"""

# Prompt 2 for AI
"""

You are a classifier that predicts the urgency level of posts in a MOOC discussion forum.

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

Choose the level that best matches the urgency implied by the post.

Here are some examples of posts and their urgency levels:

Post: "has never had anyone in it that was good at math, so you will never be a great math student either."
Urgency level: 1
Post: "See [Sir Ken Robinson's video and read his text]\date of manufacture\"  = birth date  [1]: http://blogs.edweek.org/teachers/living-in-dialogue/2011/01/sir_ken_robinson_shakes_up_the.html""
Urgency level: 1
Post: "Spent 3 mos in Finland, it was cold, windy, snowing.....even inside I was cold all the time, when everyone was in t-shirts"
Urgency level: 1
Post: "Hi! My name is Edward, I'm from MoscowIs there anybody from Russia?:)"

Post: "I agree with Chris' comment. Another way to think of this is to remember ethnicity is qualitative. If you put all three options into a single variable you're turning ethnicity into a quantitative [continuous] variable."
Urgency level: 2
Post: "I was excited to see the boys so joyful about math and that math is something they all can do.  The feeling of being free and talking so easily about their experiences since second grade.  I was surprised about them using the term decomposing so naturally and talking nonchalantly about friendly numbers such as 7 with ease.  Also that everybody makes mistakes sent a cultural shift in these young minds!"
Urgency level: 2
Post: "If mistakes are what drives learning, the teacher should lead by making some mistakes and letting the students correct the mistakes. The challenge to the students is define why the mistake is a \mistake\". The Why is where learning occurs.""
Urgency level: 2

Post: "The due date for homework in module 5 is shown at Jul 16, 2013 00:00 UTC (usually it's 19:00 on tuesdays, not midnight). Is this a typo or have I just missed the deadline by assuming it'll be at it's usual place in time?"
Urgency level: 3
Post: "oops I just clicked Misuse flag on my own comment!  My eyes aren't good enough to read that little light colored font, so I clicked on it to see what it would do!  My bad!  Old age strikes again."
Urgency level: 3
Post: "While fitting the logistic regression model, I ignored the Apps predictor from which Y is derived. That seemed like the obvious thing to do as otherwise Apps would dominate all the other predictors in terms of its significance. I just wanted to confirm this."
Urgency level: 3

Post: "do you think putting them in groups by what skills they need to master is the same as ability grouping?I also teach multi levels and it will be challenging to not group in any way, or is that what they are saying?"
Urgency level: 4
Post: "Week 1 was just a general chat though, am I missing something?"
Urgency level: 4
Post: "Dear Anne!I am sorry it seems that I am behind in my readings & posting  - I will try read and come up with questions - I work in the community in Toronto and I find the transgender population growing - I am curious to understand more about their health issues and how they intersect with women's health issues?"
Urgency level: 4
Post: "The group meeting will cover human rights and to include issues like female status, negative rights, positive rights, human rights norms, and CEDAW treaty. Would you like me to submit a group meeting report to secure score points for the course to satisfy the group meeting component?"

Post: "I have sent my Writing Assignment. In progress section I have now a gray bar. I suppose It was graded by peers review. To complete my work I must grade four others ones but I don´t know where are those writing works. They aren't in the dashboard's assessment panel section. What should I do to obtain them?Thanks for your polite attention!"
Urgency level: 5
Post: "I missed the deadline as I thought deadline is until midnight Oct. 15, Pacific time. And I am not able to submit the self assessment. Can you please open homework 3 for one more day. I just have to submit self assessment questions only. Thanks!"
Urgency level: 5
Post: "Hi,I've joined but was busy with other things for last few days. How can I make up for last 11 days?Thank you,<nameRedac_<anon_screen_name_redacted>>"
Urgency level: 5

Post: "how effective are these drugs when used empirically to treat bacterial meningitis??? And are they recommended/used in the US or UK. thank you"
Urgency level: 6
Post: "It would be nice to know how long it took you to complete each paragraph's editing, Dr. <redacted>.I know it's a lot to ask to track that, but it seems worth knowing, don't you think?"
Urgency level: 6
Post: "I also have the same problem. To me it doesn't show how many points you get on the quiz. All quizzes this week are the same. Is this for everyone?"
Urgency level: 6

Post: "What is the best way for scpd students to submit the hw? Is there an email alias one can submit to. The myStanfordConnection site (http://scpd.stanford.edu) seems down !!"
Urgency level: 7
Post: "Dear Organizer,This is to bring to you notice that Answer to Question 1 & 3 is not correct w.r.t to concept of Price Floor & Price ceiling.  Kindly correct the same & do the revaluation.Thanks<nameRedac_<anon_screen_name_redacted>> <nameRedac_<anon_screen_name_redacted>>"
Urgency level: 7
Post: "I hope any course staff member can help us to solve this confusion asap!!!"
Urgency level: 7

"""