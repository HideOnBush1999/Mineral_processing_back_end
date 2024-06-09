from flask import Blueprint, request

welcome = Blueprint('welcome', __name__, url_prefix='/welcome')

