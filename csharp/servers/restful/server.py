import flask
import waitress

app = flask.Flask(__name__)

@app.route("/add/")
def add():
    return flask.jsonify({"result": int(flask.request.args['num1']) + int(flask.request.args['num2'])})
    
@app.route("/minus/")
def minus():
    return flask.jsonify({"result": int(flask.request.args['num1']) - int(flask.request.args['num2'])})

def main():
    waitress.serve(app, host="127.0.0.1", port=50052)




if __name__ == '__main__':

    main()