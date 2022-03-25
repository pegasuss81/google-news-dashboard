from flask import Flask, render_template, request
import pandas as pd
from news_dashboard import create_df, get_representative_df

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET'])
def main():
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/dash_input', methods=['GET'])
def get_inputs():
    keyword = request.args.get('keyword')
    start = request.args.get('start')
    end = request.args.get('end')
    n_articles = request.args.get('num_recs')

    create_df(start, end, keyword)
    get_representative_df(n_articles)
    return render_template('dashboard.html', results=articles, keyword=keyword)

if __name__ == '__main__':
    app.run(debug=True)
