from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Function to read text file content
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check_plagiarism():
    try:
        file1 = request.files['file1']
        file2 = request.files['file2']

        # Ensure upload folder exists
        upload_folder = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        # Save the uploaded files
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)

        file1_path = os.path.join(upload_folder, filename1)
        file2_path = os.path.join(upload_folder, filename2)

        file1.save(file1_path)
        file2.save(file2_path)

        # Read contents
        text1 = read_file(file1_path)
        text2 = read_file(file2_path)

        # Vectorize texts and calculate similarity
        vectorizer = TfidfVectorizer().fit_transform([text1, text2])
        similarity_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
        similarity_score = round(float(similarity_matrix[0][0]) * 100, 2)

        return render_template('index.html', similarity=similarity_score)
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
