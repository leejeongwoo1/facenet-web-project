from flask import Flask, request

app = Flask(__name__)

@app.route('/lovelyz', methods=['POST'])
def predict():
    output = dict()
    output['HI'] = "Hello"

    # response = jsonify(output)
    return output

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
    # serve(app, port=5000)