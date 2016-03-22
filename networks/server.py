from flask import Flask, request, send_file
import json
import os

app = Flask(__name__)
@app.route("/sketch", methods=["POST", "GET"])
def sketch():

	# json_data = json.loads(request.data)
	# make neurel net, pass in data
	return "3"

@app.route("/")
def root():
	return send_file(os.path.abspath("../sketch.html"))

if __name__ == "__main__":
	app.run(debug=True)


