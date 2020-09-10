# imports
import pandas as pd
from Model_Random_Recc import Random_Recc
from Model_Recc import Recommendation
from Model_Recc_TFS import Recommendation_scratch
import flask
from flask import render_template, request, jsonify

app = flask.Flask(__name__)

@app.route("/", methods = ['GET'])
def predict():
    if(request.args):
        if (request.args.get('submit') == 'recco'):
            age  = request.args.get('age', None)
            description  = request.args.get('description', None)
            genre = request.args.get('genre', None)
            # m = Recommendation_scratch(age, genre, description)
            m = Recommendation(age, genre, description)
            reccs = m.get_recommendation()

            return flask.render_template('index.html', recommendation = reccs)
        
        elif (request.args.get('submit') == 'random_recc'):
            age  = request.args.get('age', None)
            genre = request.args.get('genre', None)
            
            m = Random_Recc(age, genre)
            reccs = m.get_random_reccs()
           
            return flask.render_template('index.html', recommendation = reccs)
       
    else: 
        # m = Recommendation_scratch('','','')
        m = Recommendation('','','')
        reccs = m.get_recommendation()
        return flask.render_template('index.html', recommendation = reccs)


if __name__ == '__main__':
    app.run()

