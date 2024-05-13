from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
import os
import random
import uuid
import requests
import json
from zipfile import ZipFile
import rawpy
from PIL import Image
import io
from flask import send_file
import logging
from flask_sqlalchemy import SQLAlchemy

# import database

# app = Flask(__name__)
# app.secret_key = "your_secret_key"  # Set a secret key for session security

# # Set up flask global variables
# app.config['GATEWAY_HOST'] = 'http://127.0.0.1'
# app.config['GATEWAY_PORT'] = '5000'
# app.config['EMBEDDINGS_HOST'] = 'http://127.0.0.1'
# app.config['EMBEDDINGS_PORT'] = '5001'
# app.config['DATABASE_HOST'] = 'http://127.0.0.1'
# app.config['DATABASE_PORT'] = '5002'
# app.config['MEDIA_FOLDER'] = '/home/stefan/media'














# # DB
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost:5433/sweeper'
# db = SQLAlchemy(app)
from flask import Flask, send_file
from flask_sqlalchemy import SQLAlchemy
import logging

# Create the SQLAlchemy instance
db = SQLAlchemy()








from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, Enum
import numpy as np

import datetime
import uuid

# from app import app, db




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

    return new_user

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






def create_app():
    # Create the Flask app
    app = Flask(__name__)
    app.secret_key = "your_secret_key"  # Set a secret key for session security

    # Set up flask global variables
    app.config['GATEWAY_HOST'] = 'http://127.0.0.1'
    app.config['GATEWAY_PORT'] = '5000'
    app.config['EMBEDDINGS_HOST'] = 'http://127.0.0.1'
    app.config['EMBEDDINGS_PORT'] = '5001'
    # app.config['DATABASE_HOST'] = 'http://127.0.0.1'
    # app.config['DATABASE_PORT'] = '5002'
    app.config['MEDIA_FOLDER'] = '/home/swezel/media' # TODO to value set in .env
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost:5433/sweeper'

    # Initialize the SQLAlchemy instance with the Flask app
    db.init_app(app)

    with app.app_context():
        db.create_all()

    return app
app = create_app()



@app.route('/media/<path:filename>')
def media(filename):
    # Define the directory where your images are located
    media_folder = app.config['MEDIA_FOLDER']
    # Serve the requested file from the media directory
    return send_from_directory(media_folder, filename)


@app.route('/sweep/<string:session_id>/<path:img_path_left>/<path:img_path_right>')
def sweep_session(session_id, img_path_left, img_path_right):
    if img_path_left == 'initial':
        starting_image_url = f"{app.config['DATABASE_HOST']}:{app.config['DATABASE_PORT']}/starting_image/{session_id}"
        response = requests.get(starting_image_url)
        if response.status_code == 200:
            starting_image_path = os.path.join(session_id, response.json())
            nearest_neighbor_filename = get_nearest_neighbor(session_id, starting_image_path)
            nearest_neighbor_path = os.path.join(session_id, nearest_neighbor_filename)

            return render_template(
                    'session.html',
                    session_id=session_id,
                    img_path_left=starting_image_path,
                    img_path_right=nearest_neighbor_path
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
            img_path_right=os.path.join(session_id, img_path_right)
        )
    elif img_path_right == 'endofline':
        return render_template(
            'session.html',
            session_id=session_id,
            img_path_left=os.path.join(session_id, img_path_left),
            img_path_right='endofline.jpg'
        )


    else:
        return render_template(
                'session.html',
                session_id=session_id,
                img_path_left=os.path.join(session_id, img_path_left),
                img_path_right=os.path.join(session_id, img_path_right)
            )



def get_nearest_neighbor(session_id: str, query_image_path: str) -> str:
    query_img_filename = query_image_path.split("/")[-1]
    nearest_neighbor_url = f"{app.config['DATABASE_HOST']}:{app.config['DATABASE_PORT']}/get_nearest_neighbor/{session_id}/{query_img_filename}"
    response = requests.get(nearest_neighbor_url)
    if response.status_code == 200:
        nearest_neighbor_path = response.json()
    else:
        nearest_neighbor_path = "endofline"
    return nearest_neighbor_path


def update_image_status(session_id: str, update_image_path: str, set_status_to:str = 'reviewed_discard') -> str:
    query_img_filename = update_image_path.split("/")[-1]
    update_image_status_url = f"{app.config['DATABASE_HOST']}:{app.config['DATABASE_PORT']}/update_image_status"
    
    update_data = {
        'session_id': session_id,
        'update_image_path': query_img_filename,
        'status': set_status_to
    }
    response = requests.post(update_image_status_url, json=update_data)

    return response



# TODO maybe merge image clicked and continue-clicked
@app.route('/image_clicked/<string:position>/<string:session_id>/<path:other_img_path>', methods=['POST'])
def image_clicked(position, session_id, other_img_path):
    img_path = request.form.get('img_path')
    if img_path.split("/")[-1] == 'endofline.jpg':
        _ = update_image_status(session_id, other_img_path, set_status_to='reviewed_keep')
        return redirect(url_for('overview', username='testuser'))
    _ = update_image_status(session_id, other_img_path, set_status_to='reviewed_discard')
    try:
        nearest_neighbor_path = get_nearest_neighbor(session_id, img_path)
    except UnboundLocalError:
        nearest_neighbor_path = img_path
    if position == 'left':
        return redirect(
            url_for(
                    'sweep_session',
                    session_id=session_id,
                    img_path_left=img_path.split("/")[-1],
                    img_path_right=nearest_neighbor_path
                ))
    else:
        return redirect(
            url_for(
                    'sweep_session',
                    session_id=session_id,
                    img_path_left=nearest_neighbor_path,
                    img_path_right=img_path.split("/")[-1]
                ))


@app.route('/continue_clicked/<string:position>/<string:session_id>/<path:other_img_path>', methods=['POST'])
def continue_clicked(position, session_id, other_img_path):
    img_path = request.form.get('img_path')
    _ = update_image_status(session_id, other_img_path, set_status_to='reviewed_keep')
    nearest_neighbor_path = get_nearest_neighbor(session_id, img_path)
    if position == 'left':
        return redirect(
            url_for(
                    'sweep_session',
                    session_id=session_id,
                    img_path_left=img_path.split("/")[-1],
                    img_path_right=nearest_neighbor_path
                ))
    else:
        return redirect(
            url_for(
                    'sweep_session',
                    session_id=session_id,
                    img_path_left=nearest_neighbor_path,
                    img_path_right=img_path.split("/")[-1]
                ))


@app.route('/select_seed_image', methods=['GET'])
def select_seed_image():
    logging.info("Button clicked - selecting new seed image...")
    return redirect(url_for('sweep_session'))

@app.route('/end_session', methods=['GET'])
def end_session():
    logging.info("Button clicked - returning to session overview...")
    return redirect(url_for('overview', username='testuser'))



@app.route('/overview/<string:username>')
def overview(username: str):
    """Renders an overview page listing sessions for a given user."""

    with app.app_context():
        sessions_list = get_sessions_for_user(username)
        for session in sessions_list:
            print(session.id)
            print(session.session_token)
            print(session.creation_time)
            print(session.last_access_time)



    session_images = {}  # Dictionary to store session IDs and their corresponding image paths

    # for session_id in sessions_list:
    #     get_example_image_url = f"{app.config['DATABASE_HOST']}:{app.config['DATABASE_PORT']}/get_example_images/{session_id}"
    #     image_path_response = requests.get(get_example_image_url)
    #     if image_path_response.status_code == 200:
    #         image_paths = image_path_response.json()
    #         image_paths = [os.path.join(session_id, p) for p in image_paths]
    #         # for path in image_paths:
    #         #     assert os.path.exists(path)
    #         session_images[session_id] = image_paths  # Store image paths for the session ID

    # TODO add timestamps to the overview page
    return render_template('overview.html', sessions_list=[sess.id for sess in sessions_list], session_images=[])





from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import shutil


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

# TODO remove uploading part and rename to embed_images
@app.route('/embed_images/<string:session_id>', methods=['GET', 'POST'])
def embed_images(session_id):

    image_dir = f"{app.config['MEDIA_FOLDER']}/{session_id}"  # Change this to your desired upload directory
    
    session_data = {
        'username': 'testuser',
        '_id': session_id,
        'img_dir': image_dir,
    }
    new_session_request_url = f"{app.config['DATABASE_HOST']}:{app.config['DATABASE_PORT']}/add_session"
    response = requests.post(new_session_request_url, json=session_data)

    for img_path in os.listdir(image_dir):
        # We add the jpg twin for ease of processing if the image is in raw (dng) format
        if img_path.endswith("dng") or img_path.endswith("DNG"):
            logging.info("dng detected... converting")
            display_path, download_path = convert_dng_to_jpg(os.path.join(image_dir, img_path))
        else:
            display_path, download_path = os.path.join(image_dir, img_path), os.path.join(image_dir, img_path)

        embedding_request_url = f"{app.config['EMBEDDINGS_HOST']}:{app.config['EMBEDDINGS_PORT']}/embed_image/{display_path}"
        response = requests.get(embedding_request_url)

        # Wrap embedding with additional metadata
        embedding_data = {
            'username': 'testuser',
            '_id': session_id,
            'display_path': display_path,
            'download_path': download_path,
            'embedding': response.json()
        }

        # Embeddings to to db
        insert_embedding_url = f"{app.config['DATABASE_HOST']}:{app.config['DATABASE_PORT']}/insert_embedding"
        response = requests.post(insert_embedding_url, json=embedding_data)
        if response.status_code == 200:
            logging.info(f"Image {display_path} added successfully...")
        else:
            logging.info(f"Something went wrong when attempting to add image {display_path}")
    
    # Once everything is inserted, go to overview page where added session should be listed...
    return redirect(url_for('overview', username='testuser'))


def convert_dng_to_jpg(dng_path):
    # Open the DNG file
    with rawpy.imread(dng_path) as raw:
        # Convert to RGB array
        rgb = raw.postprocess()
    
    # Create a PIL Image object from the RGB array
    img = Image.fromarray(rgb)
    # Get the directory and filename of the DNG file
    directory, filename = os.path.split(dng_path)
    # Generate the path for the JPG file in the same directory
    jpg_path = os.path.join(directory, os.path.splitext(filename)[0] + ".jpg")
    # Save the PIL Image as a JPG file
    img.save(jpg_path)
    
    return jpg_path, dng_path


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<string:session_id>', methods=['GET'])
def download_subset(session_id):
    upload_dir = os.path.join(app.config['MEDIA_FOLDER'], session_id)
    if not os.path.exists(upload_dir):
        return "Session ID not found", 404
    images_to_keep = f"{app.config['DATABASE_HOST']}:{app.config['DATABASE_PORT']}/images_to_keep/{session_id}"
    response = requests.get(images_to_keep)
    
    files = files = os.listdir(upload_dir)
    files = [file for file in files if file in response.json()]

    if not files:
        return "No files found for this session ID", 404
    # Create a zip file containing all uploaded files
    zip_filename = f"{session_id}.zip"
    zip_filepath = os.path.join(app.config['MEDIA_FOLDER'], zip_filename)
    with ZipFile(zip_filepath, 'w') as zip:
        for file in files:
            file_path = os.path.join(upload_dir, file)
            zip.write(file_path, os.path.basename(file_path))

    # Send the zip file to the client
    return send_from_directory(app.config['MEDIA_FOLDER'], zip_filename, as_attachment=True)


@app.route('/init_new_session')
def init_new_session():
    new_hash = uuid.uuid4().hex

    config = SessionConfig(
        root_dir = app.config['MEDIA_FOLDER'],
        user = "testuser",
        session_id = new_hash,
    )
    client = FileClient(config)
    client.create_dir()

    return redirect(url_for('upload_form', session_id=new_hash))


@app.route('/drop_session/<string:session_id>')
def drop_session(session_id):

    logging.info(f"Dropping session {session_id}")

    drop_session_url = f"{app.config['DATABASE_HOST']}:{app.config['DATABASE_PORT']}/drop_session"
    response = requests.post(drop_session_url, json={"username":"testuser", "_id": session_id})

    config = SessionConfig(
        root_dir = app.config['MEDIA_FOLDER'],
        user = "testuser",
        session_id = session_id,
    )
    client = FileClient(config)
    client.remove_directory()

    return redirect(url_for('overview', username='testuser'))



class SessionConfig():
    def __init__(
                self,
                root_dir: str,
                user: str,
                session_id: str,
                embeddings_api: str ="http://127.0.0.1:5050/embed",
            ):
        self.root_dir = root_dir
        self.session_id = session_id
        self.user = user
        self.embeddings_api = embeddings_api



class FileClient():
    def __init__(self, config):
        self.config = config

    def create_dir(self):
        """ Create new dir in config.root_dir with name config.session_id. """
        new_dir = os.path.join(self.config.root_dir, self.config.session_id)
        assert not os.path.exists(new_dir)
        os.mkdir(new_dir)


    def remove_directory(self):
        dir_to_remove = os.path.join(self.config.root_dir, self.config.session_id)
        zip_to_remove = os.path.join(self.config.root_dir, f"{self.config.session_id}.zip")
        assert os.path.exists(dir_to_remove)

        try:
            os.remove(zip_to_remove)
            logging.info(f"Zipfile '{zip_to_remove}' successfully removed.")
        except FileNotFoundError as e:
            logging.info(f"No file {zip_to_remove} found: {e.strerror}")

        try:
            # Iterate over all files and subdirectories in the directory
            for root, dirs, files in os.walk(dir_to_remove, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)  # Remove each file

                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)  # Remove each subdirectory

            # After all files and subdirectories are removed, remove the empty directory itself
            os.rmdir(dir_to_remove)
            logging.info(f"Directory '{dir_to_remove}' successfully removed.")
        except OSError as e:
            logging.info(f"Error: {dir_to_remove} : {e.strerror}")
    



if __name__ == '__main__':
    app.run(port=app.config['GATEWAY_PORT'],debug=True)