import re
from dateutil import parser
from dateutil.tz import gettz

def split_file(raw):
    return raw.split('/')[0]

def extract_to(raw):
    matches = re.findall(r'([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})', raw)
    if matches:
        return [email.strip() for email in matches]
    else:
        return None

def extract_date(raw):
    match = re.search(r'Date: (.+)', raw)
    if match:
        return match.group(1)
    else:
        return None

def to_utc(date):
    date = parser.parse(date)
    date_utc = date.astimezone(gettz('UTC'))
    return date_utc.timestamp()

def extract_to_org(email):
    return email.split('@')[-1]