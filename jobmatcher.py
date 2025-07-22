import boto3
import io
import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.text import MIMEText
from config import S3_BUCKET, S3_KEY, EMAIL_SENDER, EMAIL_RECEIVER, SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASS

def get_resume_text_from_s3():
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
    file_bytes = response['Body'].read()

    if S3_KEY.endswith('.pdf'):
        return extract_text(io.BytesIO(file_bytes))
    
    elif S3_KEY.endswith('.docx'):
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        return '\n'.join([para.text for para in doc.paragraphs])
    
    else:
        raise ValueError("Unsupported file type. Only .pdf and .docx are supported.")

def scrape_job_descriptions():
    job_urls = [
        "https://careers.atlassian.com/jobs",
        "https://careers.google.com/jobs/results/",
        "https://www.zomato.com/careers",
        "https://jobs.netflix.com/search",
        "https://careers.veeam.com/"
        "https://recruitment.macquarie.com/en_US/careers"
        # Add 40+ other career URLs here
    ]
    jobs = []
    keywords = ['devops', 'site reliability', 'sre', 'platform engineer']

    for url in job_urls:
        try:
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            text = soup.get_text().lower()
            if any(keyword in text for keyword in keywords):
                jobs.append((url, text))
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    return jobs

def compute_similarity(resume_text, job_text):
    vectorizer = TfidfVectorizer().fit_transform([resume_text, job_text])
    return cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0] * 100

def send_email(matches):
    body = "\n\n".join([f"{url} - {score:.2f}%" for url, score in matches])
    msg = MIMEText(body)
    msg['Subject'] = "🔍 Matched DevOps/SRE Jobs This Hour"
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(EMAIL_SENDER, [EMAIL_RECEIVER], msg.as_string())
        print("✅ Email sent!")

def main():
    resume_text = get_resume_text_from_s3()
    jobs = scrape_job_descriptions()
    matches = []

    for url, job_desc in jobs:
        similarity = compute_similarity(resume_text, job_desc)
        if similarity >= 65:
            matches.append((url, similarity))

    if matches:
        send_email(matches)
    else:
        print("No matches found this round.")

if __name__ == '_main_':
    main()