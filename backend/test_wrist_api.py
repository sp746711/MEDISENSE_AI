import getpass
import json
import requests

BASE_URL = "http://127.0.0.1:8000/api"

email = input("Enter your registered email: ").strip()
password = getpass.getpass("Enter your password: ")

session = requests.Session()

# 1. Authenticate
response = session.post(
    f"{BASE_URL}/auth/login",
    json={"email": email, "password": password},
    timeout=30,
)

print("\nLOGIN STATUS:", response.status_code)

if not response.ok:
    print(response.text)
    raise SystemExit("Login failed. Check the response above.")

data = response.json()
token = data["access_token"]
session.headers.update({"Authorization": f"Bearer {token}"})
print("Login successful.")

# 2. Create an assessment that accepts X-ray images
response = session.post(
    f"{BASE_URL}/assessments",
    json={"input_types": ["xray"]},
    timeout=30,
)

print("\nASSESSMENT STATUS:", response.status_code)
print(response.text)

if not response.ok:
    raise SystemExit("Assessment creation failed.")

assessment_id = response.json()["assessment_id"]
print("\nASSESSMENT ID:", assessment_id)
