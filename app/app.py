from typing import List, Optional
import os
import logging
import datetime
from dotenv import find_dotenv, load_dotenv

import random
import uuid
import numpy as np

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import requests
from werkzeug.utils import secure_filename

from flask_sqlalchemy import SQLAlchemy
from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, Enum

import utils

# TODOs
# TODO auth0 integration
# TODO sort out mixed use of id and session_token in database tables
# TODO get rid of unnecessary arguments for routing functions if possible
# TODO add indices to database tables
# TODO rename sessions to something else in order to avoid confusion with flask session


ENV_FILE = find_dotenv('.env.dev') # TODO make this flag dependent
if ENV_FILE:
    load_dotenv(ENV_FILE)

# Create the SQLAlchemy instance
db = SQLAlchemy() # maybe make this upper case (?)


class User(db.Model):
    __tablename__: str = 'users'
    id: int = db.Column(db.Integer, primary_key=True)
    email: str = db.Column(db.String(255), nullable=False, unique=True)
    nickname: str = db.Column(db.String(255))
    subscribed: bool = db.Column(db.Boolean, default=False)

    def __repr__(self) -> str:
        return f"User('{self.nickname}', '{self.email}')"




class Session(db.Model):
    # TODO maybe make session_token primary key?
    # TODO or maybe rename id to something else to avoid confusion?
    __tablename__ = 'sessions'
    id: int = db.Column(db.Integer, primary_key=True)
    user_id: int = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_token: str = db.Column(db.String(36), unique=True, nullable=False)
    creation_time: datetime.datetime = db.Column(db.DateTime, nullable=False, default=datetime.datetime.now())
    last_access_time: datetime.datetime = db.Column(db.DateTime, nullable=False, default=datetime.datetime.now())

    def __repr__(self) -> str:
        return f"Session('{self.session_token}', '{self.id}')"


class Embedding(db.Model):
    __tablename__ = 'embeddings'
    id: int = db.Column(db.Integer, primary_key=True)
    display_path: str = db.Column(db.String(255), nullable=False)
    download_path: str = db.Column(db.String(255), nullable=False)
    session_token: str = db.Column(db.String(36), db.ForeignKey('sessions.session_token'), nullable=False)
    embedding: np.ndarray = db.Column(Vector(384), nullable=False)
    status: str = db.Column(Enum('reviewed_keep', 'reviewed_discard', 'unreviewed', name='status'), nullable=False, default='unreviewed')

    def __repr__(self) -> str:
        return f"Embedding('{self.display_path}', '{self.download_path}', '{self.session_token}', '{self.status}')"


def add_user(email: str, nickname="", subscribed: bool = False) -> User:
    new_user = User(email=email, nickname=nickname, subscribed=subscribed)
    db.session.add(new_user)
    db.session.commit()

    return new_user



def add_session_for_user(email: str, session_token: str) -> Session:
    user = User.query.filter_by(email=email).first()
    if user:
        new_session = Session(user_id=user.id, session_token=session_token)
        db.session.add(new_session)
        db.session.commit()
        return new_session
    else:
        return None


def get_sessions_for_user(email: str) -> List[Session]:
    user = User.query.filter_by(email=email).first()
    if user:
        sessions = Session.query.filter_by(user_id=user.id).all()
        return sessions
    else:
        return None


def add_embedding_for_session(session_id: int, display_path: str, download_path: str, embedding: np.ndarray) -> Embedding:
    session = Session.query.get(session_id)
    if session:
        new_embedding = Embedding(session_token=session.session_token, display_path=display_path, download_path=download_path, embedding=embedding)
        db.session.add(new_embedding)
        db.session.commit()
        return new_embedding
    else:
        return None


def remove_session_for_user(email: str, session_token: str) -> bool:
    user = User.query.filter_by(email=email).first()
    if user:
        session = Session.query.filter_by(user_id=user.id, session_token=session_token).first()
        if session:
            # Remove all embeddings for this session
            embeddings = Embedding.query.filter_by(session_token=session.session_token).all()
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

def get_images_to_keep(session_id: str) -> List[str]:
    embeddings = Embedding.query.filter_by(session_token=session_id).all()
    images_to_keep = []
    for embedding in embeddings:
        if embedding.status == 'reviewed_keep':
            images_to_keep.append(embedding.download_path)
    return images_to_keep


def get_image_by_path(session_id: str, image_path: str) -> Embedding:
    return Embedding.query.filter_by(session_token=session_id, display_path=image_path).first()


def get_starting_image(session_id: str) -> Optional[Embedding]:
    unreviewed_images = Embedding.query.filter_by(session_token=session_id, status='unreviewed').all()
    if unreviewed_images:
        return random.choice(unreviewed_images)
    else:
        return None


def get_nearest_neighbor(session_id: str, query_image_id: int) -> Embedding:
    """ Get the nearest neighbor to the query image. """
    query_embedding = Embedding.query.get(query_image_id)
    nns = (db.session.query(Embedding)
            .filter(Embedding.session_token == session_id)
            .filter(Embedding.id != query_image_id)
            .filter(Embedding.status == 'unreviewed')
            .order_by(Embedding.embedding.l2_distance(query_embedding.embedding))
            .limit(1)
            .all())[0]
    return nns


def update_image_status(session_id: str, update_image_path: str, set_status_to:str = 'reviewed_discard') -> str:
    """ Update the status of an image in the database. """
    image = Embedding.query.filter_by(session_token=session_id, display_path=update_image_path).first()
    if image:
        image.status = set_status_to
        db.session.commit()
    else:
        logging.error(f"Image {update_image_path} not found in database.")
        return False

    return True



# App related functions and routes
def create_app():
    # Create the Flask app
    app = Flask(__name__)
    app.secret_key = "your_secret_key"  # Set a secret key for session security

    # Set up flask global variables
    app.config['GATEWAY_HOST'] = 'http://127.0.0.1'
    app.config['GATEWAY_PORT'] = '5000'
    app.config['EMBEDDINGS_HOST'] = 'http://127.0.0.1'
    app.config['EMBEDDINGS_PORT'] = '5001'
    app.config['MEDIA_FOLDER'] = os.getenv("MEDIA_FOLDER")
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URI")

    # Initialize the SQLAlchemy instance with the Flask app
    db.init_app(app)

    with app.app_context():
        db.create_all()
        # add_user('testuser@testmail.com', 'testuser')


    return app
app = create_app()


# Routes
@app.route('/media/<path:filename>')
def media(filename):
    # Define the directory where your images are located
    media_folder = app.config['MEDIA_FOLDER']
    # Serve the requested file from the media directory
    return send_from_directory(media_folder, filename)



@app.route('/sweep/<string:session_id>/left/<path:img_path_left>/right/<path:img_path_right>')
def sweep_decision(session_id, img_path_left, img_path_right): # TODO replace API calls with database queries
    if img_path_left == 'initial':
        starting_image = get_starting_image(session_id)
        if starting_image:
            nearest_neighbor = get_nearest_neighbor(session_id, starting_image.id)
            return render_template(
                    'session.html',
                    session_id=session_id,
                    img_path_left=starting_image.display_path,
                    img_path_right=nearest_neighbor.display_path
                )
        else:
            return render_template(
                    'session.html',
                    session_id=session_id,
                    img_path_left="endofline.jpg",
                    img_path_right="endofline.jpg"
                )

    elif img_path_left == 'endofline':
        return render_template(
            'session.html',
            session_id=session_id,
            img_path_left='endofline.jpg',
            img_path_right=session_id
        )
    elif img_path_right == 'endofline':
        return render_template(
            'session.html',
            session_id=session_id,
            img_path_left=session_id,
            img_path_right='endofline.jpg'
        )


    else:
        return render_template(
                'session.html',
                session_id=session_id,
                img_path_left=img_path_left,
                img_path_right=img_path_right
            )



@app.route('/image_clicked/<string:position>/<string:session_id>/clicked/<path:clicked_img_path>/other/<path:other_img_path>', methods=['POST'])
def image_clicked(position, session_id, clicked_img_path, other_img_path):
    if clicked_img_path.split("/")[-1] == 'endofline.jpg':
        _ = update_image_status(session_id, other_img_path, set_status_to='reviewed_keep')
        return redirect(url_for('overview', email='testuser@testmail.com'))

    _ = update_image_status(session_id, other_img_path, set_status_to='reviewed_discard')
    try:
        clicked_img = get_image_by_path(session_id, clicked_img_path)
        nearest_neighbor_path = get_nearest_neighbor(session_id, clicked_img.id).display_path

    except UnboundLocalError:
        nearest_neighbor_path = clicked_img_path
    if position == 'left':
        return redirect(
            url_for(
                    'sweep_decision',
                    session_id=session_id,
                    img_path_left=clicked_img_path,
                    img_path_right=nearest_neighbor_path
                ))
    else:
        return redirect(
            url_for(
                    'sweep_decision',
                    session_id=session_id,
                    img_path_left=nearest_neighbor_path,
                    img_path_right=clicked_img_path
                ))


@app.route('/continue_clicked/<string:position>/<string:session_id>/clicked/<path:clicked_img_path>/other/<path:other_img_path>', methods=['POST'])
def continue_clicked(position, session_id, clicked_img_path, other_img_path):
    _ = update_image_status(session_id, other_img_path, set_status_to='reviewed_keep')
    clicked_img = get_image_by_path(session_id, clicked_img_path)
    nearest_neighbor_path = get_nearest_neighbor(session_id, clicked_img.id).display_path

    if position == 'left':
        return redirect(
            url_for(
                    'sweep_decision',
                    session_id=session_id,
                    img_path_left=clicked_img_path,
                    img_path_right=nearest_neighbor_path
                ))
    else:
        return redirect(
            url_for(
                    'sweep_decision',
                    session_id=session_id,
                    img_path_left=nearest_neighbor_path,
                    img_path_right=clicked_img_path
                ))


@app.route('/select_seed_image', methods=['GET'])
def select_seed_image():
    logging.info("Button clicked - selecting new seed image...")
    return redirect(url_for('sweep_decision'))


@app.route('/end_session', methods=['GET'])
def end_session():
    logging.info("Button clicked - returning to session overview...")
    return redirect(url_for('overview', email='testuser@testmail.com'))


@app.route('/overview/<string:email>')
def overview(email: str):
    """Renders an overview page listing sessions for a given user."""

    with app.app_context():
        sessions_list = get_sessions_for_user(email)

    session_images = {}  # Dictionary to store session IDs and their corresponding image paths

    for session_id in sessions_list:
        # Get the image paths for the session from the database
        embeddings = Embedding.query.filter_by(session_token=session_id.session_token).limit(3).all()
        image_paths = [embedding.display_path for embedding in embeddings]
        session_images[session_id.session_token] = image_paths

    return render_template('overview.html', sessions_list=[sess.session_token for sess in sessions_list], session_images=session_images)


@app.route('/upload_form/<string:session_id>')
def upload_form(session_id):
    return render_template('upload.html', session_id=session_id)



@app.route('/upload_image/<string:session_id>', methods=['POST'])
def upload_image(session_id):
    logging.info(f"Uploading file to {app.config['MEDIA_FOLDER']}/{session_id}")
    image_dir = os.path.join(app.config['MEDIA_FOLDER'], session_id)

    if 'files' not in request.files:
        return redirect(request.url)
    file = request.files['files']
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(image_dir, filename))
    return '', 204  # Return 204 No Content response



@app.route('/upload_done/<string:session_id>', methods=['GET', 'POST'])
def upload_done(session_id):
    return f'Upload for {session_id} completed'

@app.route('/embed_images/<string:session_id>', methods=['GET', 'POST'])
def embed_images(session_id):

    image_dir = f"{app.config['MEDIA_FOLDER']}/{session_id}"
    new_session = add_session_for_user('testuser@testmail.com', session_id)

    logging.info(f"New session added with ID {new_session.id}")

    for img_path in os.listdir(image_dir):
        # We add the jpg twin for ease of processing if the image is in raw (dng) format
        if img_path.endswith(("dng", "DNG")):
            logging.info("dng detected... converting")
            display_path, download_path = utils.convert_dng_to_jpg(os.path.join(image_dir, img_path))
        else:
            display_path, download_path = os.path.join(image_dir, img_path), os.path.join(image_dir, img_path)


        # TODO replace with actual embedding from embeddings API
        # embedding_request_url = f"{app.config['EMBEDDINGS_HOST']}:{app.config['EMBEDDINGS_PORT']}/embed_image/{display_path}"
        # response = requests.get(embedding_request_url)
        embedding = np.random.rand(384)

        # Write the embedding to the database
        embedding_row = add_embedding_for_session(new_session.id, utils.strip_media_folder_from_path(app.config['MEDIA_FOLDER'], display_path), download_path, embedding)
        if embedding_row:
            logging.info(f"Image {display_path} added successfully. ..")
        else:
            logging.info(f"Something went wrong when attempting to add image {display_path}")

    # Once everything is inserted, go to overview page where added session should be listed...
    return redirect(url_for('overview', email='testuser@testmail.com'))



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<string:session_id>', methods=['GET'])
def download_subset(session_id):
    file_client = utils.FileClient(
        media_folder = app.config['MEDIA_FOLDER'],
        session_id = session_id,
    )
    # TODO make this part of the FileClient class
    upload_dir = file_client.upload_dir

    if not os.path.exists(upload_dir):
        return "Session ID not found", 404
    subset = get_images_to_keep(session_id)
    if not subset:
        # TODO send message to client that no images were selected
        return redirect(url_for('overview', email='testuser@testmail.com'))
    
    # Create a zip file containing all uploaded files
    zip_filename = file_client.zip_dir(subset)
    # Send the zip file to the client
    return send_from_directory(app.config['MEDIA_FOLDER'], zip_filename, as_attachment=True)


@app.route('/init_new_session')
def init_new_session():
    new_hash = uuid.uuid4().hex

    client = utils.FileClient(
        media_folder = app.config['MEDIA_FOLDER'],
        session_id = new_hash,
    )
    client.create_dir()

    return redirect(url_for('upload_form', session_id=new_hash))


@app.route('/drop_session/<string:session_id>')
def drop_session(session_id):
    """ Remove a session and all its contents from the database and the media directory."""
    success = remove_session_for_user('testuser@testmail.com', session_id)
    if success:
        logging.info(f"Session {session_id} successfully removed from database.")
    else:
        logging.info(f"Something went wrong when attempting to remove session {session_id}.")

    client = utils.FileClient(
        media_folder = app.config['MEDIA_FOLDER'],
        session_id = session_id,
    )
    client.remove_directory()

    return redirect(url_for('overview', email='testuser@testmail.com'))




if __name__ == '__main__':
    app.run(port=app.config['GATEWAY_PORT'],debug=True)