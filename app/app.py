from typing import List, Optional
import os
import logging
import datetime
from dotenv import find_dotenv, load_dotenv

import random
import uuid
import numpy as np
import json

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    session,
    jsonify,
)
import requests
from werkzeug.utils import secure_filename

from flask_sqlalchemy import SQLAlchemy
from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, Enum, func

from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user
from authlib.integrations.flask_client import OAuth
from urllib.parse import quote_plus, urlencode


import utils

# TODOs
# TODO sort out mixed use of id and sweep_session_token in database tables
# TODO get rid of unnecessary arguments for routing functions where possible
# TODO add indices to database tables
# TODO add login_required where suitable

ENV_FILE = find_dotenv(".env.dev")  # TODO make this flag dependent
if ENV_FILE:
    load_dotenv(ENV_FILE)

# Create the SQLAlchemy instance
db = SQLAlchemy()  # maybe make this upper case (?)


class User(db.Model):
    __tablename__: str = "users"
    id: int = db.Column(db.Integer, primary_key=True)
    email: str = db.Column(db.String(255), nullable=False, unique=True)
    nickname: str = db.Column(db.String(255))
    subscribed: bool = db.Column(db.Boolean, default=False)

    def __repr__(self) -> str:
        return f"User('{self.nickname}', '{self.email}')"


class SweepSession(db.Model):
    # TODO maybe make sweep_session_token primary key?
    # TODO or maybe rename id to something else to avoid confusion?
    __tablename__ = "sweep_sessions"
    id: int = db.Column(db.Integer, primary_key=True)
    user_id: int = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    sweep_session_token: str = db.Column(db.String(36), unique=True, nullable=False)
    creation_time: datetime.datetime = db.Column(
        db.DateTime, nullable=False, default=datetime.datetime.now()
    )
    last_access_time: datetime.datetime = db.Column(
        db.DateTime, nullable=False, default=datetime.datetime.now()
    )

    def __repr__(self) -> str:
        return f"SweepSession('{self.sweep_session_token}', '{self.id}')"


class Embedding(db.Model):
    __tablename__ = "embeddings"
    id: int = db.Column(db.Integer, primary_key=True)
    display_path: str = db.Column(db.String(255), nullable=False)
    download_path: str = db.Column(db.String(255), nullable=False)
    sweep_session_token: str = db.Column(
        db.String(36),
        db.ForeignKey("sweep_sessions.sweep_session_token"),
        nullable=False,
    )
    embedding: np.ndarray = db.Column(Vector(384), nullable=False)
    status: str = db.Column(
        Enum("reviewed_keep", "reviewed_discard", "unreviewed", name="status"),
        nullable=False,
        default="unreviewed",
    )

    def __repr__(self) -> str:
        return f"Embedding('{self.display_path}', '{self.download_path}', '{self.sweep_session_token}', '{self.status}')"



def add_user(email: str, nickname="", subscribed: bool = False) -> User:
    new_user = User(email=email, nickname=nickname, subscribed=subscribed)
    db.session.add(new_user)
    db.session.commit()

    return new_user


def get_user(email: str) -> User:
    return User.query.filter_by(email=email).first()


def add_session_for_user(email: str, sweep_session_token: str) -> SweepSession:
    user = User.query.filter_by(email=email).first()
    if user:
        new_sweep_session = SweepSession(
            user_id=user.id, sweep_session_token=sweep_session_token
        )
        db.session.add(new_sweep_session)
        db.session.commit()
        return new_sweep_session
    else:
        return None


def get_sessions_for_user(email: str) -> List[SweepSession]:
    user = User.query.filter_by(email=email).first()
    if user:
        sessions = SweepSession.query.filter_by(user_id=user.id).all()
        return sessions
    else:
        return None


def add_embedding_for_sweep_session(
    sweep_session_id: int, display_path: str, download_path: str, embedding: np.ndarray
) -> Embedding:
    session = SweepSession.query.get(sweep_session_id)
    if session:
        new_embedding = Embedding(
            sweep_session_token=session.sweep_session_token,
            display_path=display_path,
            download_path=download_path,
            embedding=embedding,
        )
        db.session.add(new_embedding)
        db.session.commit()
        return new_embedding
    else:
        return None


def remove_session_for_user(email: str, sweep_session_token: str) -> bool:
    user = User.query.filter_by(email=email).first()
    if user:
        session = SweepSession.query.filter_by(
            user_id=user.id, sweep_session_token=sweep_session_token
        ).first()
        if session:
            # Remove all embeddings for this session
            embeddings = Embedding.query.filter_by(
                sweep_session_token=session.sweep_session_token
            ).all()
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


def get_images_to_keep(sweep_session_id: str) -> List[str]:
    embeddings = Embedding.query.filter_by(sweep_session_token=sweep_session_id).all()
    images_to_keep = []
    for embedding in embeddings:
        if embedding.status == "reviewed_keep":
            images_to_keep.append(embedding.download_path)
    return images_to_keep


def get_image_by_path(sweep_session_id: str, image_path: str) -> Embedding:
    return Embedding.query.filter_by(
        sweep_session_token=sweep_session_id, display_path=image_path
    ).first()


def get_starting_image(sweep_session_id: str) -> Optional[Embedding]:
    unreviewed_images = Embedding.query.filter_by(
        sweep_session_token=sweep_session_id, status="unreviewed"
    ).all()
    if unreviewed_images:
        return random.choice(unreviewed_images)
    else:
        return None


def get_nearest_neighbor(sweep_session_id: str, query_image_id: int) -> Embedding:
    """ Get the nearest neighbor to the query image. """
    query_embedding = Embedding.query.get(query_image_id)
    nns = (
        db.session.query(Embedding)
        .filter(Embedding.sweep_session_token == sweep_session_id)
        .filter(Embedding.id != query_image_id)
        .filter(Embedding.status == "unreviewed")
        .order_by(Embedding.embedding.l2_distance(query_embedding.embedding))
        .limit(1)
        .all()
    )[0]
    return nns


def update_image_status(
    sweep_session_id: str,
    update_image_path: str,
    set_status_to: str = "reviewed_discard",
) -> str:
    """ Update the status of an image in the database. """
    image = Embedding.query.filter_by(
        sweep_session_token=sweep_session_id, display_path=update_image_path
    ).first()
    if image:
        image.status = set_status_to
        db.session.commit()
    else:
        logging.error(f"Image {update_image_path} not found in database.")
        return False

    return True


def get_percentage_reviewed(sweep_session_id: str) -> int:
    count_all = len(
        Embedding.query.filter(
            Embedding.sweep_session_token == sweep_session_id
        ).all()
    )
    count_reviewed = len(
        Embedding.query.filter(
            Embedding.sweep_session_token == sweep_session_id,
            Embedding.status.in_(["reviewed_keep", "reviewed_discard"]),
        ).all()
    )
    try:
        percentage_reviewed = (count_reviewed / count_all) * 100
        return percentage_reviewed

    except ZeroDivisionError:
        return 0


# App related functions and routes
def create_app():
    # Create the Flask app
    app = Flask(__name__)
    app.secret_key = "your_secret_key"  # Set a secret key for session security

    # Set up flask global variables
    app.config["GATEWAY_HOST"] = "http://127.0.0.1"
    app.config["GATEWAY_PORT"] = "5000"
    app.config["EMBEDDINGS_HOST"] = "http://127.0.0.1"
    app.config["EMBEDDINGS_PORT"] = "5001"
    app.config["MEDIA_FOLDER"] = os.getenv("MEDIA_FOLDER")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URI")

    # Initialize the SQLAlchemy instance with the Flask app
    db.init_app(app)

    with app.app_context():
        db.create_all()

    return app


app = create_app()

# User management
oauth = OAuth(app)
login_manager = LoginManager()


class FlaskUser(UserMixin):
    def __init__(self, id):
        self.id = id


@login_manager.user_loader
def load_user(user_id):
    return FlaskUser(user_id)


oauth.register(
    "auth0",
    client_id=os.getenv("AUTH0_CLIENT_ID"),
    client_secret=os.getenv("AUTH0_CLIENT_SECRET"),
    client_kwargs={"scope": "openid profile email",},
    server_metadata_url=f'https://{os.getenv("AUTH0_DOMAIN")}/.well-known/openid-configuration',
)
login_manager.init_app(app)


@app.route("/login")
def login():
    return oauth.auth0.authorize_redirect(
        redirect_uri=url_for("callback", _external=True)
    )


@app.route("/callback", methods=["GET", "POST"])
def callback():
    token = (
        oauth.auth0.authorize_access_token()
    )  # token from auth0 contains the all the user info
    session["user"] = token
    user_email = session["user"]["userinfo"]["name"]
    user_nickname = session["user"]["userinfo"]["nickname"]

    # Check if user exists in the database
    user = get_user(user_email)
    # If user doesn't exist, create a new user
    if not user:
        add_user(user_email, user_nickname)
        logging.info(f"User {user_email} added to database.")

    login_user(FlaskUser(user_email))
    return redirect("/overview")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(
        "https://"
        + os.getenv("AUTH0_DOMAIN")
        + "/v2/logout?"
        + urlencode(
            {
                "returnTo": url_for("home", _external=True),
                "client_id": os.getenv("AUTH0_CLIENT_ID"),
            },
            quote_via=quote_plus,
        )
    )


# Routes (non-user management related)
@app.route("/")
def home():
    return render_template(
        "home.html",
        session=session.get("user"),
        pretty=json.dumps(session.get("user"), indent=4),
    )


@app.route("/profile")
@login_required
def profile():
    user_info = session.get("userinfo")
    user_info = session.get("user")

    return render_template("profile.html", user_info=user_info)


@app.route("/media/<path:filename>")
def media(filename):
    # Define the directory where your images are located
    media_folder = app.config["MEDIA_FOLDER"]
    # Serve the requested file from the media directory
    return send_from_directory(media_folder, filename)


@app.route(
    "/sweep/<string:sweep_session_id>/left/<path:img_path_left>/right/<path:img_path_right>"
)
# @login_required
# def sweep_decision(sweep_session_id, img_path_left, img_path_right):
#     if img_path_left == "initial":
#         starting_image = get_starting_image(sweep_session_id)
#         if starting_image:
#             nearest_neighbor = get_nearest_neighbor(sweep_session_id, starting_image.id)
#             return render_template(
#                 "sweep_session.html",
#                 sweep_session_id=sweep_session_id,
#                 img_path_left=starting_image.display_path,
#                 img_path_right=nearest_neighbor.display_path,
#             )
#         else:
#             return render_template(
#                 "sweep_session.html",
#                 sweep_session_id=sweep_session_id,
#                 img_path_left="endofline.jpg",
#                 img_path_right="endofline.jpg",
#             )

#     elif img_path_left == "endofline":
#         return render_template(
#             "sweep_session.html",
#             sweep_session_id=sweep_session_id,
#             img_path_left="endofline.jpg",
#             img_path_right=sweep_session_id,
#         )
#     elif img_path_right == "endofline":
#         return render_template(
#             "sweep_session.html",
#             sweep_session_id=sweep_session_id,
#             img_path_left=sweep_session_id,
#             img_path_right="endofline.jpg",
#         )

#     else:
#         return render_template(
#             "sweep_session.html",
#             sweep_session_id=sweep_session_id,
#             img_path_left=img_path_left,
#             img_path_right=img_path_right,
#         )
def sweep_decision(sweep_session_id, img_path_left, img_path_right):
    if img_path_left == "initial":
        starting_image = get_starting_image(sweep_session_id)
        if starting_image:
            nearest_neighbor = get_nearest_neighbor(sweep_session_id, starting_image.id)
            return render_template(
                "decision.html",
                sweep_session_id=sweep_session_id,
                img_path_left=starting_image.display_path,
                img_path_right=nearest_neighbor.display_path,
            )
        else:
            return render_template(
                "decision.html",
                sweep_session_id=sweep_session_id,
                img_path_left="endofline.jpg",
                img_path_right="endofline.jpg",
            )

    elif img_path_left == "endofline":
        return render_template(
            "decision.html",
            sweep_session_id=sweep_session_id,
            img_path_left="endofline.jpg",
            img_path_right=sweep_session_id,
        )
    elif img_path_right == "endofline":
        return render_template(
            "decision.html",
            sweep_session_id=sweep_session_id,
            img_path_left=sweep_session_id,
            img_path_right="endofline.jpg",
        )

    else:
        return render_template(
            "decision.html",
            sweep_session_id=sweep_session_id,
            img_path_left=img_path_left,
            img_path_right=img_path_right,
        )



@login_required
@app.route('/like_image', methods=['POST'])
def like_image():
    image_src = request.json.get('imageSrc')
    print(f"Image liked: {image_src}")
    # Add your logic here to handle the liked image
    return jsonify({'status': 'success', 'imageSrc': image_src})


@login_required
@app.route('/drop_image', methods=['POST'])
def drop_image():
    image_src = request.json.get('imageSrc')
    print(f"Image dropped: {image_src}")
    # Add your logic here to handle the dropped image
    return jsonify({'status': 'success', 'imageSrc': image_src})


@login_required
@app.route('/continue_from', methods=['POST'])
def continue_from():
    image_src = request.json.get('imageSrc')
    other_image_src = request.json.get('otherImageSrc')
    print(f"Continue from: {image_src}, dropping {other_image_src}")
    # Add your logic here to continue from the selected image
    return jsonify({'status': 'success', 'imageSrc': image_src})
                   

# TODO remove this route (replace with like_image, drop_image, continue_from)
@app.route(
    "/image_clicked/<string:position>/<string:sweep_session_id>/clicked/<path:clicked_img_path>/other/<path:other_img_path>",
    methods=["POST"],
)
@login_required
def image_clicked(position, sweep_session_id, clicked_img_path, other_img_path):
    if clicked_img_path.split("/")[-1] == "endofline.jpg":
        _ = update_image_status(
            sweep_session_id, other_img_path, set_status_to="reviewed_keep"
        )
        return redirect(url_for("overview"))

    _ = update_image_status(
        sweep_session_id, other_img_path, set_status_to="reviewed_discard"
    )
    try:
        clicked_img = get_image_by_path(sweep_session_id, clicked_img_path)
        nearest_neighbor_path = get_nearest_neighbor(
            sweep_session_id, clicked_img.id
        ).display_path

    except UnboundLocalError:
        nearest_neighbor_path = clicked_img_path
    if position == "left":
        return redirect(
            url_for(
                "sweep_decision",
                sweep_session_id=sweep_session_id,
                img_path_left=clicked_img_path,
                img_path_right=nearest_neighbor_path,
            )
        )
    else:
        return redirect(
            url_for(
                "sweep_decision",
                sweep_session_id=sweep_session_id,
                img_path_left=nearest_neighbor_path,
                img_path_right=clicked_img_path,
            )
        )


# TODO remove this route (replace with like_image, drop_image, continue_from)
@app.route(
    "/continue_clicked/<string:position>/<string:sweep_session_id>/clicked/<path:clicked_img_path>/other/<path:other_img_path>",
    methods=["POST"],
)
@login_required
def continue_clicked(position, sweep_session_id, clicked_img_path, other_img_path):
    _ = update_image_status(
        sweep_session_id, other_img_path, set_status_to="reviewed_keep"
    )
    clicked_img = get_image_by_path(sweep_session_id, clicked_img_path)
    nearest_neighbor_path = get_nearest_neighbor(
        sweep_session_id, clicked_img.id
    ).display_path

    if position == "left":
        return redirect(
            url_for(
                "sweep_decision",
                sweep_session_id=sweep_session_id,
                img_path_left=clicked_img_path,
                img_path_right=nearest_neighbor_path,
            )
        )
    else:
        return redirect(
            url_for(
                "sweep_decision",
                sweep_session_id=sweep_session_id,
                img_path_left=nearest_neighbor_path,
                img_path_right=clicked_img_path,
            )
        )


@app.route("/select_seed_image", methods=["GET"])
def select_seed_image():
    logging.info("Button clicked - selecting new seed image...")
    return redirect(url_for("sweep_decision"))


@app.route("/end_session", methods=["GET"])
def end_session():
    logging.info("Button clicked - returning to session overview...")
    return redirect(url_for("overview"))


@app.route("/overview")
@login_required
def overview():
    """Renders an overview page listing sessions for a given user."""
    with app.app_context():
        sweep_sessions_list = get_sessions_for_user(
            session.get("user")["userinfo"]["name"]
        )

    sweep_session_images = (
        {}
    )  # Dictionary to store session IDs and their corresponding image paths
    sweep_session_progress_percentage = {}

    for sweep_session in sweep_sessions_list:
        # Get the image paths for the session from the database
        embeddings = (
            Embedding.query.filter_by(
                sweep_session_token=sweep_session.sweep_session_token
            )
            .limit(3)
            .all()
        )
        image_paths = [embedding.display_path for embedding in embeddings]
        sweep_session_images[sweep_session.sweep_session_token] = image_paths
        percentage = get_percentage_reviewed(sweep_session.sweep_session_token)
        sweep_session_progress_percentage[
            sweep_session.sweep_session_token
        ] = percentage

    return render_template(
        "overview.html",
        sweep_sessions_list=[sess.sweep_session_token for sess in sweep_sessions_list],
        sweep_session_images=sweep_session_images,
        sweep_session_progress_percentage=sweep_session_progress_percentage,
    )


@app.route("/upload_form/<string:sweep_session_id>")
@login_required
def upload_form(sweep_session_id):
    return render_template("upload.html", sweep_session_id=sweep_session_id)


@app.route("/upload_image/<string:sweep_session_id>", methods=["POST"])
def upload_image(sweep_session_id):
    logging.info(f"Uploading file to {app.config['MEDIA_FOLDER']}/{sweep_session_id}")
    image_dir = os.path.join(app.config["MEDIA_FOLDER"], sweep_session_id)

    if "files" not in request.files:
        return redirect(request.url)
    file = request.files["files"]
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(image_dir, filename))
    return "", 204  # Return 204 No Content response


@app.route("/upload_done/<string:sweep_session_id>", methods=["GET", "POST"])
def upload_done(sweep_session_id):
    return f"Upload for {sweep_session_id} completed"


@app.route("/embed_images/<string:sweep_session_id>", methods=["GET", "POST"])
def embed_images(sweep_session_id):

    image_dir = f"{app.config['MEDIA_FOLDER']}/{sweep_session_id}"

    new_sweep_session = add_session_for_user(
        session.get("user")["userinfo"]["name"], sweep_session_id
    )

    logging.info(f"New session added with ID {new_sweep_session.id}")

    for img_path in os.listdir(image_dir):
        # We add the jpg twin for ease of processing if the image is in raw (dng) format
        if img_path.endswith(("dng", "DNG")):
            logging.info("dng detected... converting")
            display_path, download_path = utils.convert_dng_to_jpg(
                os.path.join(image_dir, img_path)
            )
        else:
            display_path, download_path = (
                os.path.join(image_dir, img_path),
                os.path.join(image_dir, img_path),
            )

        # # TODO replace with actual embedding from embeddings API
        # embedding_request_url = f"{app.config['EMBEDDINGS_HOST']}:{app.config['EMBEDDINGS_PORT']}/embed_image/{display_path}"
        # response = requests.get(embedding_request_url)
        # embedding = response.json()

        embedding = np.random.rand(384)

        # Write the embedding to the database
        embedding_row = add_embedding_for_sweep_session(
            new_sweep_session.id,
            utils.strip_media_folder_from_path(
                app.config["MEDIA_FOLDER"], display_path
            ),
            download_path,
            embedding,
        )
        if embedding_row:
            logging.info(f"Image {display_path} added successfully. ..")
        else:
            logging.info(
                f"Something went wrong when attempting to add image {display_path}"
            )

    # Once everything is inserted, go to overview page where added session should be listed...
    return redirect(url_for("overview"))


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/download/<string:sweep_session_id>", methods=["GET"])
def download_subset(sweep_session_id):
    file_client = utils.FileClient(
        media_folder=app.config["MEDIA_FOLDER"], sweep_session_id=sweep_session_id,
    )
    # TODO make this part of the FileClient class
    upload_dir = file_client.upload_dir

    if not os.path.exists(upload_dir):
        return "SweepSession ID not found", 404
    subset = get_images_to_keep(sweep_session_id)
    if not subset:
        # TODO send message to client that no images were selected
        return redirect(url_for("overview"))

    # Create a zip file containing all uploaded files
    zip_filename = file_client.zip_dir(subset)
    # Send the zip file to the client
    return send_from_directory(
        app.config["MEDIA_FOLDER"], zip_filename, as_attachment=True
    )


@app.route("/init_new_sweep_session")
def init_new_sweep_session():
    new_hash = uuid.uuid4().hex

    client = utils.FileClient(
        media_folder=app.config["MEDIA_FOLDER"], sweep_session_id=new_hash,
    )
    client.create_dir()

    return redirect(url_for("upload_form", sweep_session_id=new_hash))


@app.route("/drop_sweep_session/<string:sweep_session_id>")
def drop_sweep_session(sweep_session_id):
    """ Remove a session and all its contents from the database and the media directory."""
    success = remove_session_for_user(
        session.get("user")["userinfo"]["name"], sweep_session_id
    )
    if success:
        logging.info(
            f"SweepSession {sweep_session_id} successfully removed from database."
        )
    else:
        logging.info(
            f"Something went wrong when attempting to remove session {sweep_session_id}."
        )

    client = utils.FileClient(
        media_folder=app.config["MEDIA_FOLDER"], sweep_session_id=sweep_session_id,
    )
    client.remove_directory()

    return redirect(url_for("overview"))


if __name__ == "__main__":
    app.run(port=app.config["GATEWAY_PORT"], debug=True)
