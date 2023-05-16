from pyspark.sql.functions import split, explode
from functools import reduce
from pyspark.sql.functions import sum as _sum
from math import log
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import enchant
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from sklearn.metrics import confusion_matrix


stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer(language='english')
english_dictionary = enchant.Dict("en_US")

LAPLACE_SMOOTHING_PARAMETER = 1
FREQUENCY_THRESHOLD = 10

def calPriorProbs(data):
    counts = data.groupBy("sentiment").count().rdd.collectAsMap()
    total = counts["positive"] + counts["negative"]
    return counts["positive"]/total, counts["negative"]/total

def processReview(review):
    return [stemmer.stem(word) for word in review.split() if word.strip() != '' and word not in stop_words and english_dictionary.check(word)]

def postProcessing(words_counts):
    processed_words_counts = {}

    for word in words_counts.keys():
        if word.strip() != '' and word not in stop_words and english_dictionary.check(word):
            processed_words_counts[stemmer.stem(word)] = words_counts[word] + processed_words_counts.get(stemmer.stem(word), 0)

    return {key:value for key, value in processed_words_counts.items() if value > FREQUENCY_THRESHOLD}
    
def countWordsInAClass(reviews_data_frame, class_label):
    class_reviews = reviews_data_frame.filter(reviews_data_frame.sentiment == class_label)
    words_column = explode(split(class_reviews.review, "\s+")).alias("word")
    words_counts = class_reviews.select(words_column).groupBy("word").count()
    total_count = words_counts.agg(_sum("count")).collect()[0][0]
    words_counts = words_counts.rdd.collectAsMap()
    return {"total-count":total_count, "words-counts":postProcessing(words_counts)}
    
def train(data):
    pos_prior_prob, neg_prior_prob = calPriorProbs(data)
    pos_counts = countWordsInAClass(data, "positive")
    neg_counts = countWordsInAClass(data, "negative")
    parameters = {
        "pos-prior-prob":pos_prior_prob,
        "neg-prior-prob":neg_prior_prob,
        "pos-counts":pos_counts,
        "neg-counts":neg_counts}
    return parameters

def calLogProb(words, class_counts, class_prior_prob):
    probs_list = [class_counts["words-counts"].get(word, LAPLACE_SMOOTHING_PARAMETER)/(class_counts["total-count"] + LAPLACE_SMOOTHING_PARAMETER*len(class_counts["words-counts"])) for word in words]
    return log(class_prior_prob) + reduce(lambda a, b: a + log(b), probs_list, 0)

def predict(review, parameters):
    processed_words = processReview(review)
    pos_log_prob = calLogProb(processed_words, parameters["pos-counts"], parameters["pos-prior-prob"])
    neg_log_prob = calLogProb(processed_words, parameters["neg-counts"], parameters["neg-prior-prob"])
    return "positive" if pos_log_prob > neg_log_prob else "negative"

def calAccuracy(data, parameters):
    predictor_udf = udf(lambda review: predict(review, parameters), StringType())
    predictions = data.select('sentiment', predictor_udf(data.review).alias('prediction'))
    accuracy = predictions.filter(predictions.sentiment == predictions.prediction).count()/predictions.count()
    return accuracy

def calPrecision(data, parameters):
    predictor_udf = udf(lambda review: predict(review, parameters), StringType())
    predictions = data.select('sentiment', predictor_udf(data.review).alias('prediction'))
    true_positives = predictions.filter((predictions.sentiment == 'positive') & (predictions.prediction == 'positive')).count()
    predicted_positives = predictions.filter(predictions.prediction == 'positive').count()
    precision = true_positives / predicted_positives
    return precision

def calRecall(data, parameters):
    predictor_udf = udf(lambda review: predict(review, parameters), StringType())
    predictions = data.select('sentiment', predictor_udf(data.review).alias('prediction'))
    true_positives = predictions.filter((predictions.sentiment == 'positive') & (predictions.prediction == 'positive')).count()
    actual_positives = predictions.filter(predictions.sentiment == 'positive').count()
    recall = true_positives / actual_positives
    return recall

def calF1Score(data, parameters):
    predictor_udf = udf(lambda review: predict(review, parameters), StringType())
    predictions = data.select('sentiment', predictor_udf(data.review).alias('prediction'))
    
    true_positives = predictions.filter((predictions.sentiment == 'positive') & (predictions.prediction == 'positive')).count()
    false_positives = predictions.filter((predictions.sentiment == 'negative') & (predictions.prediction == 'positive')).count()
    false_negatives = predictions.filter((predictions.sentiment == 'positive') & (predictions.prediction == 'negative')).count()

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score


def generateConfusionMatrix(data, parameters):
    predictor_udf = udf(lambda review: predict(review, parameters), StringType())
    predictions = data.select('sentiment', predictor_udf(data.review).alias('prediction'))

    true_labels = predictions.select('sentiment').rdd.flatMap(lambda x: x).collect()
    predicted_labels = predictions.select('prediction').rdd.flatMap(lambda x: x).collect()

    cm = confusion_matrix(true_labels, predicted_labels)
    return cm