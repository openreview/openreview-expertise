import openreview
import os
import sys

def generate_token():
    try:
        client = openreview.api.OpenReviewClient(
            "http://localhost:3001", 
            username="openreview.net", 
            password="Or$3cur3P@ssw0rd"
        )
        with open("api_token.txt", "w") as f:
            f.write(client.token)
        print("Token generated successfully")
        return 0
    except Exception as e:
        print(f"Error generating token: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(generate_token())