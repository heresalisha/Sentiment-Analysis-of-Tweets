from flask import Flask, render_template, request, url_for
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

app = Flask(__name__, template_folder="template")
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Load sentiment analysis model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

labels = ["Negative", "Neutral", "Positive"]


def analyze_tweet_inputs(tweet):
    # Preprocess user input
    tweet_words = []
    for word in tweet.split(" "):
        if word.startswith("@") and len(word) > 1:
            word = "@user"
        elif word.startswith("http"):
            word = "http"
        tweet_words.append(word)

    tweet_proc = " ".join(tweet_words)
    # Sentiment analysis
    encoded_tweet = tokenizer(tweet_proc, return_tensors="pt")
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Get the sentiment label with the highest probability
    predicted_sentiment_index = scores.argmax()
    predicted_sentiment = labels[predicted_sentiment_index]

    # Calculate the percentage
    percentage = round(scores[predicted_sentiment_index] * 100, 2)

    return {"sentiment": predicted_sentiment, "percentage": percentage}


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/analyze", methods=["POST", "GET"])
def analyze():
    if request.method == "GET":
        return render_template("one_tweet.html")
    user_input = request.form["tweet"]

    result = analyze_tweet_inputs(user_input)
    predicted_sentiment = result["sentiment"]
    percentage = result["percentage"]



    image_url = ""
    if predicted_sentiment == "Neutral":
        image_url = url_for("static", filename="neu.png")
    elif predicted_sentiment == "Positive":
        image_url = url_for("static", filename="pos.jpg")
    else:
        image_url = url_for("static", filename="neg.jpg")

    # # Calculate the percentage
    # percentage = round(scores[predicted_sentiment_index] * 100, 2)

    return render_template(
        "result.html",
        sentiment=predicted_sentiment,
        percentage=percentage,
        image_url=image_url,
    )


@app.route("/more_analyze", methods=["POST", "GET"])
def more_analyze():
    if request.method == "GET":
        return render_template("more_tweet.html")
    if request.method == "POST":
        user_input1 = request.form["t1"]
        user_input2 = request.form["t2"]
        user_input3 = request.form["t3"]
        user_input4 = request.form["t4"]
        user_input5 = request.form["t5"]

    tweets = [user_input1, user_input2, user_input3, user_input4, user_input5]

    all_results = []
    for tweet in tweets:
        result = analyze_tweet_inputs(tweet)
        result["tweet"] = tweet
        all_results.append(result)

    for res in all_results:
        print(res)

    return render_template("result_multiple.html", data=all_results)


if __name__ == "__main__":
    app.run(debug=True)
