from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, Enum
import numpy as np

import datetime
import uuid





class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), nullable=False, unique=True)
    email = db.Column(db.String(255), nullable=False)
    subscribed = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"



class Session(db.Model):
    __tablename__ = 'sessions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_token = db.Column(db.String(36), unique=True, nullable=False)
    creation_time = db.Column(db.DateTime, nullable=False, default=datetime.datetime.now())
    last_access_time = db.Column(db.DateTime, nullable=False, default=datetime.datetime.now())

    def __repr__(self):
        return f"Session('{self.session_token}', '{self.id}')"


class Embedding(db.Model):
    __tablename__ = 'embeddings'
    id = db.Column(db.Integer, primary_key=True)
    display_path = db.Column(db.String(255), nullable=False)
    download_path = db.Column(db.String(255), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey('sessions.id'), nullable=False)
    embedding = db.Column(Vector(384), nullable=False)
    status = db.Column(Enum('reviewed_keep', 'reviewed_discard', 'unreviewed', name='status'), nullable=False, default='unreviewed')

    def __repr__(self):
        return f"Embedding('{self.display_path}', '{self.download_path}', '{self.session_id}', '{self.status}')"





def add_user(username, email, subscribed=False):
    new_user = User(username=username, email=email, subscribed=subscribed)
    db.session.add(new_user)
    db.session.commit()





def add_session_for_user(username):
    user = User.query.filter_by(username=username).first()
    if user:
        session_token = str(uuid.uuid4().hex)
        new_session = Session(user_id=user.id, session_token=session_token)
        db.session.add(new_session)
        db.session.commit()
        return new_session
    else:
        return None


def get_sessions_for_user(username):
    user = User.query.filter_by(username=username).first()
    if user:
        sessions = Session.query.filter_by(user_id=user.id).all()
        return sessions
    else:
        return None

def add_embedding_for_session(session_id, display_path, download_path, embedding):
    session = Session.query.get(session_id)
    if session:
        new_embedding = Embedding(session_id=session.id, display_path=display_path, download_path=download_path, embedding=embedding)
        db.session.add(new_embedding)
        db.session.commit()
        return new_embedding
    else:
        return None



def remove_session_for_user(username, session_id):
    user = User.query.filter_by(username=username).first()
    if user:
        session = Session.query.get(session_id)
        if session:
            # Remove all embeddings for this session
            embeddings = Embedding.query.filter_by(session_id=session.id).all()
            for embedding in embeddings:
                db.session.delete(embedding)
                db.session.commit()
            # Remove the session
            db.session.delete(session)
            db.session.commit()

            return True
        else:
            return False
    else:
        return False

if __name__ == '__main__':
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost:5433/sweeper'

    db = SQLAlchemy(app)
    with app.app_context():
        db.create_all()
        
        # add_user('testuser', 'testuser@example.com')
        users = User.query.all()
        for user in users:
            print(user.username)

        # new_sess = add_session_for_user(users[0].username)
        # List sessions
        sessions = get_sessions_for_user('testuser')
        for session in sessions:
            print(session.id)
            print(session.session_token)


        for i in range(5):
            add_embedding_for_session(sessions[0].id, f'/path/to/display/{i:04}.jpg', f'/path/to/download/{i:04}.dng', np.random.rand(384))



        # remove_session_for_user('testuser', sessions[0].id)
        # sessions = get_sessions_for_user('testuser')
        # Remove
        # remove_session_for_user('testuser', sessions[0].id)




# index = Index(
#     'my_index',
#     Embedding.embedding,
#     postgresql_using='hnsw',
#     postgresql_with={'m': 16, 'ef_construction': 64},
#     postgresql_ops={'embedding': 'vector_l2_ops'}
# )
