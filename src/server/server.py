from flask import Flask, request, render_template


# declare a flask app initialized to a Flask instance
app = Flask(__name__)

# define an endpoint to return the main html form
@app.rout("/", methods=["GET"])
def index():
  return render_template("index.html")


# define an endpoint to handle predictions
@app.route("/predict", methods=["POST"])
def predict():
  return render_template("predicted.html")


if __name__ == "__main__":
  # run() method of Flask class runs the application on the local development server
  app.run()
