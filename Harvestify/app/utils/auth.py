from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash

def register_user(mongo, username, password):
    hashed_password = generate_password_hash(password)
    if mongo.db.users.find_one({"username": username}):
        return False  # Username already exists
    mongo.db.users.insert_one({"username": username, "password": hashed_password})
    return True

def authenticate_user(mongo, username, password):
    user = mongo.db.users.find_one({"username": username})
    if user and check_password_hash(user["password"], password):
        return user
    return None
