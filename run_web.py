"""Run the Sawtooth web app from the PROJECT ROOT:  python run_web.py"""
from web import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # SupplyMind: http://127.0.0.1:5001
