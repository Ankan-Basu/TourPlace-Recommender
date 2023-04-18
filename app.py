from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

x = ['Kolkata', 'Delhi']

@app.route('/', methods=['GET'])
def hello():
    # print('Inside')
    return jsonify(message=x)

@app.route('/echo', methods=['POST'])
def echo():
    data = request.json
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
