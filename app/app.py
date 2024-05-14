from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, send_file
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



from flask_sqlalchemy import SQLAlchemy
from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, Enum
import numpy as np

import datetime
import uuid

# Create the SQLAlchemy instance
db = SQLAlchemy()




# DB related
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), nullable=False, unique=True)
    email = db.Column(db.String(255), nullable=False)
    subscribed = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"



class Session(db.Model):
    # TODO maybe make session_token primary key?
    # TODO or maybe rename id to something else to avoid confusion?
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
    session_token = db.Column(db.String(36), db.ForeignKey('sessions.session_token'), nullable=False)
    embedding = db.Column(Vector(384), nullable=False)
    status = db.Column(Enum('reviewed_keep', 'reviewed_discard', 'unreviewed', name='status'), nullable=False, default='unreviewed')

    def __repr__(self):
        return f"Embedding('{self.display_path}', '{self.download_path}', '{self.session_token}', '{self.status}')"





def add_user(username, email, subscribed=False):
    new_user = User(username=username, email=email, subscribed=subscribed)
    db.session.add(new_user)
    db.session.commit()

    return new_user

def add_session_for_user(username, session_token):
    user = User.query.filter_by(username=username).first()
    if user:
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
        new_embedding = Embedding(session_token=session.session_token, display_path=display_path, download_path=download_path, embedding=embedding)
        db.session.add(new_embedding)
        db.session.commit()
        return new_embedding
    else:
        return None



def remove_session_for_user(username, session_token):
    user = User.query.filter_by(username=username).first()
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



def get_images_to_keep(session_id):
    # Get 
    embeddings = Embedding.query.filter_by(session_token=session_id).all()
    images_to_keep = []
    for embedding in embeddings:
        if embedding.status == 'reviewed_keep':
            images_to_keep.append(embedding.download_path)
    return images_to_keep


def get_image_by_path(session_id, image_path):
    return Embedding.query.filter_by(session_token=session_id, display_path=image_path).first()


def get_starting_image(session_id):
    """ Get a random unreviewed image from the session. """
    unreviewed_images = Embedding.query.filter_by(session_token=session_id, status='unreviewed').all()
    # unreviewed_images = [embedding.display_path for embedding in embeddings]
    if unreviewed_images:
        return random.choice(unreviewed_images)
    else:
        return None


def get_nearest_neighbor(session_id, query_image_id):
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
    # update_image_path = add_media_folder_to_path(update_image_path)
    # Update status of the image in the database
    image = Embedding.query.filter_by(session_token=session_id, display_path=update_image_path).first()
    if image:
        image.status = set_status_to
        db.session.commit()
    else:
        logging.error(f"Image {update_image_path} not found in database.")
        return False

    return True




# TODO remove to utils
def strip_media_folder_from_path(path):
    return path.replace(app.config['MEDIA_FOLDER'] + '/', '')

def add_media_folder_to_path(path):
    return os.path.join(app.config['MEDIA_FOLDER'], path)





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
        return redirect(url_for('overview', username='testuser'))
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


# @app.route('/continue_clicked/<string:position>/<string:session_id>/<path:other_img_path>', methods=['POST'])
@app.route('/continue_clicked/<string:position>/<string:session_id>/clicked/<path:clicked_img_path>/other/<path:other_img_path>', methods=['POST'])
def continue_clicked(position, session_id, clicked_img_path, other_img_path):
    img_path = request.form.get('img_path')
    _ = update_image_status(session_id, other_img_path, set_status_to='reviewed_keep')
    # nearest_neighbor_path = get_nearest_neighbor(session_id, img_path)
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
    return redirect(url_for('overview', username='testuser'))



@app.route('/overview/<string:username>')
def overview(username: str):
    """Renders an overview page listing sessions for a given user."""

    with app.app_context():
        sessions_list = get_sessions_for_user(username)

    session_images = {}  # Dictionary to store session IDs and their corresponding image paths

    for session_id in sessions_list:
        # Get the image paths for the session from the database
        embeddings = Embedding.query.filter_by(session_token=session_id.session_token).limit(5).all()
        image_paths = [embedding.display_path for embedding in embeddings]
        session_images[session_id.session_token] = image_paths

    # TODO add timestamps to the overview page
    return render_template('overview.html', sessions_list=[sess.session_token for sess in sessions_list], session_images=session_images)





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

    image_dir = f"{app.config['MEDIA_FOLDER']}/{session_id}"
    new_session = add_session_for_user('testuser', session_id)

    logging.info(f"New session added with ID {new_session.id}")

    for img_path in os.listdir(image_dir):
        # We add the jpg twin for ease of processing if the image is in raw (dng) format
        if img_path.endswith(("dng", "DNG")):
            logging.info("dng detected... converting")
            display_path, download_path = convert_dng_to_jpg(os.path.join(image_dir, img_path))
        else:
            display_path, download_path = os.path.join(image_dir, img_path), os.path.join(image_dir, img_path)


        # TODO replace with actual embedding from embeddings API
        # embedding_request_url = f"{app.config['EMBEDDINGS_HOST']}:{app.config['EMBEDDINGS_PORT']}/embed_image/{display_path}"
        # response = requests.get(embedding_request_url)
        embedding = np.random.rand(384)

        # Write the embedding to the database
        embedding_row = add_embedding_for_session(new_session.id, strip_media_folder_from_path(display_path), download_path, embedding)
        if embedding_row:
            logging.info(f"Image {display_path} added successfully. ..")
        else:
            logging.info(f"Something went wrong when attempting to add image {display_path}")

    # Once everything is inserted, go to overview page where added session should be listed...
    return redirect(url_for('overview', username='testuser'))




# TODO move to utils
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
    subset = get_images_to_keep(session_id)
    if not subset:
        # TODO send message to client that no images were selected
        return redirect(url_for('overview', username='testuser'))
    
    # Create a zip file containing all uploaded files
    zip_filename = f"{session_id}.zip"
    zip_filepath = os.path.join(app.config['MEDIA_FOLDER'], zip_filename)
    with ZipFile(zip_filepath, 'w') as zip:
        for file in subset:
            # file_path = os.path.join(app.config['MEDIA_FOLDER'], file)
            
            zip.write(file, os.path.basename(file))

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

    # Remove the session from the database
    success = remove_session_for_user('testuser', session_id)
    if success:
        logging.info(f"Session {session_id} successfully removed from database.")
    else:
        logging.info(f"Something went wrong when attempting to remove session {session_id}.")


    # drop_session_url = f"{app.config['DATABASE_HOST']}:{app.config['DATABASE_PORT']}/drop_session"
    # response = requests.post(drop_session_url, json={"username":"testuser", "_id": session_id})



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