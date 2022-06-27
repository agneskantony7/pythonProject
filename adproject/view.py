from flask import Blueprint, render_template
view = Blueprint('view', __name__)


@view.route(methods=['POST'])
def home():
    pass
    return "<h1>hgjyft</h1>"
    #return render_template("ad.html")