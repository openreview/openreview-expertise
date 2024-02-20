import argparse
from expertise.service.server import app
import os
from expertise.service import load_model_artifacts
import threading

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default=5000, type=int)
    parser.add_argument('--container', action='store_true')
    parser.add_argument('--block_on_artifacts', action='store_true')
    args = parser.parse_args()

    if args.container:
        if args.block_on_artifacts: ## Batch prediction does not call /startup
            load_model_artifacts()
        else:
            threading.Thread(target=load_model_artifacts).start()


    app.run(host=args.host, port=args.port)