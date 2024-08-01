# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Expose the port the app runs on
EXPOSE 5000

# Set an environment variable based on the MODE argument
ARG MODE
ENV MODE=${MODE}

# Print the MODE value to verify it's set correctly
RUN echo "MODE is set to ${MODE}"

# Set the default command to run based on the mode
CMD if [ "$MODE" = "development" ]; then \
        export FLASK_ENV=development && export FLASK_DEBUG=1 && flask run --host=0.0.0.0; \
    else \
        gunicorn --bind 0.0.0.0:5000 app:app; \
    fi