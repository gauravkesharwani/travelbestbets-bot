import chatter4
import logging
from flask import Flask, render_template, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


CORS(app)
app.static_folder = 'static'

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s'

)

logger = logging.getLogger(__name__)


class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    question = db.Column(db.String(256))
    answer = db.Column(db.String(1000))

    def to_dict(self):
        return {
            'id': self.id,
            'date': self.date,
            'question': self.question,
            'answer': self.answer
        }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat")
def get_bot_response():
    userText = request.args.get('msg')

    logger.debug("Conversation Customer:" + userText)

    response = chatter4.get_response(userText)

    logger.debug("Conversation Chatbot: " + response)

    conversation = Conversation(question=userText,
                                answer=response)

    db.session.add(conversation)
    db.session.commit()

    return response


@app.route('/history')
def history():
    return render_template('chat_history.html')


@app.route('/api/data')
def data():
    query = Conversation.query

    # search filter
    search = request.args.get('search')
    if search:
        query = query.filter(db.or_(
            Conversation.question.like(f'%{search}%'),
            Conversation.answer.like(f'%{search}%')
        ))
    total = query.count()

    # sorting
    sort = request.args.get('sort')
    if sort:
        order = []
        for s in sort.split(','):
            direction = s[0]
            name = s[1:]
            if name not in ['date', 'question', 'answer']:
                name = 'date'
            col = getattr(Conversation, name)
            if direction == '-':
                col = col.desc()
            order.append(col)
        if order:
            query = query.order_by(*order)

    # pagination
    start = request.args.get('start', type=int, default=-1)
    length = request.args.get('length', type=int, default=-1)
    if start != -1 and length != -1:
        query = query.offset(start).limit(length)

    # response
    return {
        'data': [conversation.to_dict() for conversation in query],
        'total': total,
    }


if __name__ == "__main__":
    app.run(debug=True)
