import openreview
import os
import sys
import json

# Helper script to generate a token for the container tests
# and populate an example JSON file with the token

# Writes out the test JSON to test_input.json

def generate_token(path_to_json):
    try:
        client = openreview.api.OpenReviewClient(
            "http://localhost:3001", 
            username="openreview.net", 
            password="Or$3cur3P@ssw0rd"
        )

        # Load JSON, write token to dict and write JSON to disk
        with open(path_to_json, "r") as f:
            data = json.load(f)
        data["token"] = client.token
        with open("test_input.json", "w") as f:
            json.dump(data, f)

        print("Token generated successfully")
        return 0
    except Exception as e:
        print(f"Error generating token: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(generate_token(sys.argv[1]))