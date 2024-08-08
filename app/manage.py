from flask.cli import FlaskGroup
from flask import session, g
from app import app, db, User
import os

cli = FlaskGroup(app)

@cli.command("create_db")
def create_db():
    db.drop_all()
    db.create_all()
    user1 = User(email="testuser@testmail.com", nickname="testuser", subscribed=True)
    db.session.add(user1)
    db.session.commit()

@cli.command("seed_db")
def seed_db():
    ...

# @app.before_request
# def clear_session_in_development():
#     if os.getenv('FLASK_ENV') == 'development':
#         session.clear()
#     g.user = None

if __name__ == "__main__":
    cli()