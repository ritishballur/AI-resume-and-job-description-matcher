from flask import Flask, request, jsonify
from matcher import Matcher
from utils import extract_text_from_pdf

app = Flask(__name__)
matcher = Matcher(model_name='all-MiniLM-L6-v2')

@app.route('/match', methods=['POST'])
def match():
    data = request.get_json()
    resume = data.get('resume_text', '')
    job = data.get('job_text', '')
    top_k = int(data.get('top_k', 5))

    if not resume or not job:
        return jsonify({'error': 'resume_text and job_text required'}), 400

    results = matcher.match_resume_to_jd(resume, job, top_k=top_k)
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)