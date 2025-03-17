'''from flask import Flask, render_template, request, redirect, url_for, flash
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = '2003'

# Connect to MongoDB (replace with your MongoDB URI)
client = MongoClient('mongodb+srv://thehara00:Z9WsYmoMkid1OHC9@featherfind.6qqef.mongodb.net/')  # Use your MongoDB URI here
db = client['user_database']
users_collection = db['users']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Validate that username and password are not empty
        if not username or not password:
            flash('Username and password are required!', 'danger')
            return render_template('register.html')

        # Check if the username is already taken
        if users_collection.find_one({'username': username}):
            flash('Username already exists! Please choose another one.', 'danger')
            return render_template('register.html')

        # Hash the password before saving it
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Insert the new user into the database
        users_collection.insert_one({'username': username, 'password': hashed_password})
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))  # Redirect to the login page after successful registration

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Validate that username and password are not empty
        if not username or not password:
            flash('Username and password are required!', 'danger')
            return render_template('login.html')

        # Check if the user exists in the database
        user = users_collection.find_one({'username': username})

        # If user exists, verify the password
        if user:
            if check_password_hash(user['password'], password):  # Verify the hashed password
                flash('Login successful!', 'success')
                return redirect(url_for('home'))  # Redirect to the homepage after successful login
            else:
                flash('Incorrect password, please try again.', 'danger')
        else:
            flash('Username not found, please try again or register.', 'danger')

    return render_template('login.html')

if __name__ == "__main__":
    app.run(debug=True)
'''
from flask import Flask, render_template, request, redirect, url_for, flash
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import re  # Import for email validation

app = Flask(__name__)
app.secret_key = '2003'

# Connect to MongoDB (replace with your MongoDB URI)
client = MongoClient('mongodb+srv://thehara00:Z9WsYmoMkid1OHC9@featherfind.6qqef.mongodb.net/')  # Use your MongoDB URI here
db = client['user_database']
users_collection = db['users']

def is_valid_email(email):
    """Validate email format."""
    email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(email_regex, email)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Validate username/email is not empty
        if not username or not password:
            flash('Username and password are required!', 'danger')
            return render_template('register.html')

        # Check if username is an email and validate it
        if '@' in username:
            if not is_valid_email(username):
                flash('Invalid email format!', 'danger')
                return render_template('register.html')

        # Validate password length
        if len(password) < 3:
            flash('Password must be at least 3 characters long!', 'danger')
            return render_template('register.html')

        # Check if the username/email already exists
        if users_collection.find_one({'username': username}):
            flash('Username or email already exists! Please choose another one.', 'danger')
            return render_template('register.html')

        # Hash the password before saving it
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Insert the new user into the database
        users_collection.insert_one({'username': username, 'password': hashed_password})
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))  # Redirect to the login page after successful registration

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Validate that username/email and password are not empty
        if not username or not password:
            flash('Username and password are required!', 'danger')
            return render_template('login.html')

        # Allow login with either email or username
        user = users_collection.find_one({'username': username})

        if user and check_password_hash(user['password'], password):
            flash('Login successful!', 'success')
            return redirect(url_for('home'))  # Redirect to the homepage after successful login
        else:
            flash('Invalid username/email or password.', 'danger')

    return render_template('login.html')

if __name__ == "__main__":
    app.run(debug=True)
