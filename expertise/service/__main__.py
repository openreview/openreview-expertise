import argparse
from expertise.service.server import app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default=5000, type=int)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)