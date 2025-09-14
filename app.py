from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from flask_mail import Mail
import requests, csv, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

app = Flask(__name__)
app.secret_key = "secret-key"

# ---------------------------
# Database setup
# ---------------------------
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ---------------------------
# Mail setup
# ---------------------------
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your_app_password'
mail = Mail(app)

# ---------------------------
# User model
# ---------------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    mobile = db.Column(db.String(20), nullable=False)
    roll_no = db.Column(db.String(50), nullable=False)
    college = db.Column(db.String(150), nullable=False)
    class_name = db.Column(db.String(50), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------------------------
# Semantic model
# ---------------------------
model = SentenceTransformer("all-MiniLM-L12-v2")

# ---------------------------
# RapidAPI credentials
# ---------------------------
RAPID_API_KEY = "993fba48e9mshbde4683173e2b8cp1826c9jsn4b4926fea284"
RAPID_API_HOST = "jsearch.p.rapidapi.com"

# ---------------------------
# Job fetching & matching
# ---------------------------
def fetch_jobs_from_api(description, industry, experience):
    query = description
    if industry:
        query += f" {industry}"
    if experience:
        query += f" {experience} level"

    url = f"https://{RAPID_API_HOST}/search"
    headers = {"X-RapidAPI-Key": RAPID_API_KEY, "X-RapidAPI-Host": RAPID_API_HOST}
    querystring = {"query": query, "page": "1", "num_pages": "1", "date_posted": "all", "country": "in", "language": "en"}

    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        print("Error fetching data:", response.text)
        return []

def match_jobs_semantic(user_input, jobs, title_key="job_title"):
    if not jobs:
        return []
    titles = [job.get(title_key, "") for job in jobs]
    user_emb = model.encode([user_input])
    job_embs = model.encode(titles)
    scores = cosine_similarity(user_emb, job_embs)[0]
    for i, job in enumerate(jobs):
        job["match_score"] = round(float(scores[i]) * 100, 2)
    jobs.sort(key=lambda x: x["match_score"], reverse=True)
    return jobs

def fetch_nco_data(description):
    df = pd.read_csv("nco_2015_occupations.csv")
    nco_jobs = df.to_dict(orient="records")
    for job in nco_jobs:
        job["job_title"] = job.get("Occupation_Title", "")
        job["employer_name"] = "NCO"
        job["job_city"] = job.get("Industry", "")
        job["job_publisher"] = "NCO Database"
        job["job_apply_link"] = ""
    return nco_jobs

def save_to_csv(jobs):
    filename = "job_results.csv"
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Title", "Company", "Location", "Source", "Match Score", "Link"])
        for job in jobs:
            writer.writerow([job.get("job_title",""), job.get("employer_name",""), job.get("job_city",""),
                             job.get("job_publisher",""), job.get("match_score",""), job.get("job_apply_link","")])
    return filename

# ---------------------------
# Role check first
# ---------------------------
@app.route("/")
def root():
    return redirect(url_for("check_role"))

@app.route("/check_role")
def check_role():
    return render_template("check_role.html")

@app.route("/role_response/<choice>")
def role_response(choice):
    if choice == "yes":
        return redirect(url_for("home"))
    else:
        return render_template("no_access.html")

# ---------------------------
# Home page
# ---------------------------
@app.route("/home")
def home():
    return render_template("home.html")

# ---------------------------
# Index (after login)
# ---------------------------
@app.route("/index")
@login_required
def index():
    return render_template("index.html", username=current_user.username)

# ---------------------------
# Search
# ---------------------------
@app.route("/search", methods=["POST"])
@login_required
def search():
    description = request.form.get("description", "").strip()
    career_goal = request.form.get("career_goal", "")
    industry = request.form.get("industry", "").strip()
    experience = request.form.get("experience", "")

    if not description:
        flash("Please enter a job description or skills.", "danger")
        return render_template("index.html")

    jobs_api = fetch_jobs_from_api(description, industry, experience)
    matched_jobs_api = match_jobs_semantic(description, jobs_api, title_key="job_title")
    for job in matched_jobs_api:
        job["source"] = "API"

    nco_jobs = fetch_nco_data(description)
    matched_nco_jobs = match_jobs_semantic(description, nco_jobs, title_key="job_title")
    for job in matched_nco_jobs:
        job["source"] = "NCO"

    combined_jobs = matched_jobs_api + matched_nco_jobs
    combined_jobs.sort(key=lambda x: x["match_score"], reverse=True)
    top_jobs = combined_jobs[:5]
    save_to_csv(top_jobs)

    return render_template("results.html", jobs=top_jobs, career_goal=career_goal, description=description, username=current_user.username)

# ---------------------------
# Login
# ---------------------------
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash("Welcome back!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid credentials", "danger")
    return render_template("login.html")

# ---------------------------
# Register
# ---------------------------
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = bcrypt.generate_password_hash(request.form["password"]).decode("utf-8")
        confirm = request.form["confirm"]
        mobile = request.form["mobile"]
        roll_no = request.form["roll_no"]
        college = request.form["college"]
        class_name = request.form["class_name"]


        if request.form["password"] != confirm:
            flash("Passwords do not match", "danger")
            return redirect(url_for("register"))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered. Please login.", "warning")
            return redirect(url_for("login"))

        user = User(username=username, email=email, password=password, mobile=mobile,roll_no=roll_no,college=college,class_name=class_name)
        db.session.add(user)
        db.session.commit()

        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

# ---------------------------
# Profile
# ---------------------------
@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html", user=current_user)

# ---------------------------
# Logout
# ---------------------------
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

# ---------------------------
# Translate API
# ---------------------------
@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        translated_text = GoogleTranslator(source="auto", target="en").translate(text)
        return jsonify({"text": translated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)

