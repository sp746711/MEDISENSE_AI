
import getpass
import sys

import requests

BASE_URL = "http://127.0.0.1:8000/api"


def main():
    """Run an interactive API smoke test against a running backend."""
    email = input("Enter your registered email: ").strip()
    password = getpass.getpass("Enter your password: ")

    if not email or not password:
        print("Email and password are required.")
        return 1

    session = requests.Session()

    try:
        # 1. Authenticate
        response = session.post(
            f"{BASE_URL}/auth/login",
            json={"email": email, "password": password},
            timeout=30,
        )

        print("\nLOGIN STATUS:", response.status_code)

        if not response.ok:
            print(response.text)
            print("Login failed. Check the response above.")
            return 1

        data = response.json()
        token = data.get("access_token")

        if not token:
            print("Login response did not contain an access token.")
            return 1

        session.headers.update(
            {"Authorization": f"Bearer {token}"}
        )
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
            print("Assessment creation failed.")
            return 1

        assessment_id = response.json().get("assessment_id")

        if not assessment_id:
            print("Response did not contain an assessment_id.")
            return 1

        print("\nASSESSMENT ID:", assessment_id)
        return 0

    except requests.RequestException as exc:
        print(f"API request failed: {exc}")
        return 1
    except (ValueError, TypeError) as exc:
        print(f"Unexpected API response: {exc}")
        return 1
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
