import chatter
import logging
from flask import Flask, render_template, request


app = Flask(__name__)
app.static_folder = 'static'

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s'

)

logger = logging.getLogger(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    logger.debug("Conversation Customer:" + userText)

    response = chatter.get_response(userText)
    logger.debug("Conversation Chatbot: " + response)

    return response


if __name__ == "__main__":
    app.run(debug=True)
