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



def eval_on_csv(model, csv_path, text_col="Text", label_col="Label"):
    df = pd.read_csv(csv_path)
    X = df[text_col].astype(str)
    y = df[label_col].astype(int)

    y_pred = model.predict(X.tolist())

    print("accuracy:", accuracy_score(y, y_pred))
    print("f1:", f1_score(y, y_pred, average="binary"))
    print(classification_report(y, y_pred))


# LR_classifier = joblib.load("LR_classifier.pkl")
# print(LR_classifier.predict(["This is a test sentence."]))
# eval_on_csv(LR_classifier, "sampled_100.csv")

# Prompt for AI
"""

"""