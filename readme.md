# DDoS-classification-algorithm
DDoS机器学习分类模型

## Files
The repository is organised as follows:
- `data_processing.py/` 数据清洗，合并7个csv文件
- `select_labels.py/` 利用selectKBest筛选features
- `create_balanced_dataset.py/` 建立平衡数据集
- `balanced_dataset.csv/` 平衡数据集 
- `ml.py/` 机器学习算法分析
- `Evaluation of Classification algorithms for Distributed Denial of Service Attack Detection.pdf/` 参考论文
- `论文翻译及实验结果.pdf/` 参考论文的个人翻译版本，以及复现算法结果

## Data
数据集 https://www.unb.ca/cic/datasets/ddos-2019.html
CSV-03-11.zip中，共7个csv文件：
- LDAP.csv
- MSSQL.csv
- NetBIOS.csv
- Portmap.csv
- Syn.csv
- UDP.csv
- UDPLag.csv