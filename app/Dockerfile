# Use the official Python image from the Docker Hub
FROM python:3.11.3-slim-buster

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y netcat

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Expose the port the app runs on
EXPOSE 5000

# Set an environment variable based on the MODE argument
ARG MODE
ENV MODE=${MODE}

# Print the MODE value to verify it's set correctly
RUN echo "MODE is set to ${MODE}"

# Ensure the media directory exists
RUN mkdir -p /app/media

# Set the default command to run based on the mode
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]