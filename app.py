
# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask
from flask import render_template
from flask import request
import predictsentimence

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__,static_folder='D:\Project\WEB APP\staticFiles')

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return render_template('home.html')

# @app.route('/findyoursentimence/<text>',methods=['GET'])

# def findyoursentimence(text):
#     score = predictsentimence.predictsentimence(text)
#     return score

@app.route('/findyoursentimence', methods=['POST', 'GET'])
def findyoursentimence():
  if request.method == "POST":
    text = request.get_json()
    score = predictsentimence.predictsentimence(text)
    return score

# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application
    # on the local development server.
    app.run()