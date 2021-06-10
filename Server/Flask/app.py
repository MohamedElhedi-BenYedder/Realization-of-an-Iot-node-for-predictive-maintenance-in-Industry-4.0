from flask import Flask ,jsonify,render_template
from flask_cors import CORS, cross_origin
import pandas as pd
from pandas.core.frame import DataFrame
import requests as rqs
import mpld3
import Config
import DW
import EDA
import MS
import time 

def toHeaderHtml(df):
    columns = df
    if(isinstance(df,pd.DataFrame)):
        columns = columns = df.columns.tolist();
    row_header = '<div class="row header">'
    for col in  columns:
        row_header +='<div class="cell">'+col+'</div>'
    row_header+='</div>'
    return row_header

def dataframeToHtlm(df):
    columns = columns = df.columns.tolist();
    row_header = '<div class="row header">'
    for col in  columns:
        row_header +='<div class="cell">'+col+'</div>'
    row_header+='</div>'
    row_body = ''
    values = df.values.tolist()
    oneRow=''
    for row in values:
        index = 0
        oneRow='<div class="row">'
        for col in columns:
            oneRow+='<div class="cell" data-title="'+col+'">'+str(row[index])+'</div>'
            index+=1
        oneRow+='</div>'
        row_body+=oneRow
    return row_header+row_body
DW
EDA

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)
@app.route('/')
def main():
    return render_template("index.html")

@app.route('/Approach')
def approach():
    return render_template("approach.html")

@app.route('/DataWrangling')
def dataWrangling():
    return render_template("data_wrangling.html")

@app.route('/ExploratoryDataAnalysis')
def exploratoryDataAnalysis():
    return render_template("exploratory_data_analysis.html")

@app.route('/ModelSelection')
def modelSelection():
    return render_template("model_selection.html")
@app.route('/Prediction')
def prediction():
    return render_template("prediction.html")
"""

"""
@app.route('/engine/ConnectedDevices')
def connectedDevices():
    apiUrl = Config.api_server_main_url
    enginesUrl = apiUrl+'/engine'
    r_engines = rqs.get(enginesUrl)
    data_engines = pd.json_normalize(r_engines.json())
    connected = data_engines[data_engines['connected']==True]
    connected.drop(columns='connected',inplace=True)
    return jsonify(dataframeToHtlm(connected))
@app.route('/DW')
def show():
    return "hello "
@app.route('/DW/data')
def loadData():
    loadedData = DW.data
    return jsonify(dataframeToHtlm(loadedData))
@app.route('/DW/period')
def Period():
    period = DW.period
    return jsonify(period)
@app.route('/DW/columns_to_treat')
def treatColumns():
    treatedColumns = DW.columns_to_treat
    return jsonify(toHeaderHtml(treatedColumns))
CORS(app)

@app.route('/DW/df_fx')
@cross_origin()
def featureExtration():
    df_fx = DW.df_fx
    return jsonify(dataframeToHtlm(df_fx))
@app.route('/DW/df_labels')
@cross_origin()
def featureLabels():
    df_labels = DW.df_labels
    return jsonify(dataframeToHtlm(df_labels))


@app.route('/EDA')
@app.route('/EDA/features')
def features():
    f=EDA.features_to_study
    return jsonify(toHeaderHtml(f))
@app.route('/EDA/corrWithTTF')
def coorWithTTF():
    return jsonify(dataframeToHtlm(EDA.cor_ttf))
@app.route('/EDA/correlationMatrix')
def correlationMatrix():
    return EDA.fig_html

@app.route('/MS')
@app.route('/MS/ttf')
def ttf():
    ttf=MS.score_ttf
    return jsonify(dataframeToHtlm(ttf))
@app.route('/MS/bnc')
def bnc():
    bnc=MS.score_bnc
    return jsonify(dataframeToHtlm(bnc))
@app.route('/MS/mcc')
def mcc():
    mcc=MS.score_mcc
    return jsonify(dataframeToHtlm(mcc))
@app.route('/MS/selectedModels')
def selectedModels():
    return jsonify(toHeaderHtml(MS.SelectedModels))
@app.route('/MS/predicitions')
def predictions():
    return jsonify(dataframeToHtlm(MS.predictions))


if __name__ =="__main__" :
    app.run(debug=True)