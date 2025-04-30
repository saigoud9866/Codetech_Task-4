# Codetech_Task-4
PROJECT ON PYTHON LEARNING

Email Spam Classifier
This Python script uses machine learning to classify emails as spam or not spam (ham). It fetches emails from an IMAP server (e.g., Gmail) and predicts whether each email is spam based on its content.

Features
✔ Spam Detection – Uses a trained Logistic Regression model with TF-IDF text vectorization.
✔ IMAP Email Fetching – Connects to your email inbox and retrieves recent messages.
✔ Probability Scores – Shows the likelihood (%) of an email being spam.
✔ Preview & Summary – Displays email subjects and a short preview.

Installation
Install Required Libraries

sh
pip install numpy pandas scikit-learn imaplib2
Run the Script

sh
python spam_classifier.py
Usage
Enter your email address and password when prompted.

The script will fetch recent emails and classify them.

Results will show:

Email ID

Subject

Spam Status (SPAM or NOT SPAM)

Spam Probability (e.g., 95.2%)

Text Preview (first 100 characters)

Example Output
Email Spam Classification Results:
--------------------------------------------------
ID: 12345
Subject: "Win a Free iPhone!"
Status: SPAM (98.5%)
Preview: Congratulations! You've won a free iPhone. Claim now...
--------------------------------------------------
ID: 12346
Subject: "Meeting Tomorrow"
Status: NOT SPAM (2.1%)
Preview: Hi team, let's meet tomorrow at 10 AM...
--------------------------------------------------
Found 1 spam emails out of 10
Customization
Change IMAP Server: Modify imap_server='imap.gmail.com' for different providers (e.g., Outlook: imap-mail.outlook.com).

Adjust Spam Threshold: Modify bool(prediction) to use a probability threshold (e.g., proba[1] > 0.9 for stricter filtering).

Security Note
🔒 Avoid entering passwords directly in scripts for production use.
🔒 Use OAuth or app-specific passwords for better security.
