from pyspark.sql import SparkSession
from IMDBMovieReviewData import IMDBMovieReviewData
from multinomialNaiveBayes import train, calAccuracy, calPrecision, calRecall, calF1Score, generateConfusionMatrix
import json


spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()


print("Loading data...")
data = IMDBMovieReviewData(spark, "IMDB Dataset.csv")
data.loadData()

print("Preprocessing data...")
data.preprocessData()

print("Splitting data into training and testing set...")
data.splitData()


print("Training multinomial naive bayes on IMBD data...")
parameters = train(data.training_data)

print("Writing learned parameters to a json file...")
json_parameters = json.dumps(parameters)
with open("parameters.json", "w") as outfile:
    outfile.write(json_parameters)


# print("Calculating train accuracy...")
# print(f"Train Accuracy: {calAccuracy(data.training_data, parameters)}")

print("Calculating test accuracy...")
print(f"Test Accuracy: {calAccuracy(data.testing_data, parameters)}")

print("Calculating precision...")
print(f"Precision: {calPrecision(data.testing_data, parameters)}")

print("Calculating Recall...")
print(f"Recall: {calRecall(data.testing_data, parameters)}")

print("Calculating F1 Score...")
print(f"F1 Score: {calF1Score(data.testing_data, parameters)}")

print("Generating confusion matrix...")
print(f"Confusion Matrix: {generateConfusionMatrix(data.testing_data, parameters)}")