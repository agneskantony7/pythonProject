from flask import Blueprint, render_template
auth = Blueprint('auth')


@auth.route('/login')
def login():
    return render_template("login.html")

