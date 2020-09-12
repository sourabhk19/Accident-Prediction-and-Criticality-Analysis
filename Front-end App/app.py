from flask import Flask, render_template, request, redirect

app = Flask(__name__)#static_folder = "graphic_data")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction_home")
def prediction_home():
    return render_template("prediction_home_index.html")

@app.route("/accident_prediction")
def accident_prediction():
    return render_template("accident_prediction_index.html")

@app.route("/data_analysis_home")
def data_analysis_home():
    return render_template("data_analysis_home_index.html")

@app.route("/data_analysis")
def data_analysis():
    return render_template("data_analysis_index.html")

@app.route("/graph_image")
def graph_image():
    return render_template("graph_image.html")

@app.route("/weather_API")
def weather_API():
    return render_template("weather_API_index.html")

@app.route("/about_Team")
def about_Team():
    return render_template("about_Team_index.html")

@app.route("/accident_criticality")
def accident_criticality():
    return redirect("http://localhost:5000/")

# @app.errorhandler(404)
# def page_not_found(e):
#     return render_template('404.html'), 404

if __name__ == "__main__":
    app.run(port=5001, debug=True)