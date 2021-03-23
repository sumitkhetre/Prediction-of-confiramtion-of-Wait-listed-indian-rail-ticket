import app_ml
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def root():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    from_1 = int(request.form.get('from'))
    to_1 = int(request.form.get('to'))

    travelClass = int(request.form.get('travelClass'))
    journeyDate = request.form.get('journeyDate')

    print(request.form)

    confirm = app_ml.predict(from_1,to_1,travelClass ,journeyDate)

    if confirm==1:
        confirm='Ticket Will   get Confirm'
    else:
        confirm = 'Ticket Will  not get Confirm'


    return render_template("result.html", confirm=confirm)
    # return str(charges)


app.run(host='0.0.0.0', port=4000, debug=True)