import pandas as pd
import re

import re

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


df = pd.read_excel("stanford_dataset.xlsx")
df = df[["Text", "Urgency(1-7)"]]
df["Text"] = df["Text"].apply(clean_text)
df.rename(columns={"Urgency(1-7)": "Label"}, inplace=True)
df["Label"] = (df["Label"] >= 4).astype(int)
df.to_csv("data_processed.xlsx", index=False, encoding="utf-8-sig")
print("数据清洗并保存成功！")
