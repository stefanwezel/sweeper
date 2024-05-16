# 🧹 Sweeper
This repository contains the front, gateway and most of the backend for the sweeper image sorting app.

# Installation (under construction 🚧)
We will go over some basic steps to get a local version of the app running. This is only temporary and meant for early stage development. 


## Pre-requisites
- `Postgresql` instance with a database called `sweeper`
	- Download and installation instructions can be found here: https://www.postgresql.org/download/ 
- Add (and install) the `pgvector` extension (see https://github.com/pgvector/pgvector)
- `.env` file defning a media folder, database-uri, and authenticication variables.

## Get-started
- Clone this repository
- Get a fresh Python 3.12 or similar environment (using i.e. `conda`)
- Navigate to `sweeper` project directory
- Move your `.env` file here and rename it to `.env.dev`
- Install dependencies with `pip install -r requirements.txt`
- Navigate to `sweeper/app`
- Run `python app.py`
	- this will create the tables
- In the browser, navigate to [the landing page](127.0.0.1:5000) to check that the app is running
- Connect to your `Postgresql` instance (i.e. through `sudo -u postgres psql`)
- Seed the table users with an example user by running
	```sql
	insert into users (email, nickname) values ('testuser@testmail.com', 'testuser');
	```
- Now, you should be able to navigate to an [example overview page](http://127.0.0.1:5000/overview)
- 😺 Happy development!