Sentiment Analysis Using PySpark:

Environment Required to run:

Google Colab (easier to install specific versions)

1. For downloading IMDB Dataset.csv, refer to : https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

2. Installation Steps:

    1. !pip3 install pyspark==3.2.4 nltk pyenchant

    2. !wget -q https://dlcdn.apache.org/spark/spark-3.2.4/spark-3.2.4-bin-hadoop3.2.tgz && \
        tar -xzf spark-3.2.4-bin-hadoop3.2.tgz && \
        mv spark-3.2.4-bin-hadoop3.2 spark && \
        rm spark-3.2.4-bin-hadoop3.2.tgz

    3. !apt-get install -y enchant

3. Training and Evaluation of Naive Bayes Classifier:

        !python main.py

4. Please refer to source_code/532_project.ipynb for Demo, Evaluation and Test cases.

