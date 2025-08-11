Pedict apple stock price with LSTM and delpoy using fastapi

Data is downloaded from Kaggle. Apple stock price is extracted and explored. I tried to use
LSTM and LightGBM. LightGBM give almost constant prediction at the end. So I choose to
use LSTM. The model is evaluated using RSME on the test data. After that trianed model
and scaler are saved for building the application.

FastAPI is used to depoly the application as an API. 

run vscode as admin
ctrl+shft+P WSL:remote
open the folder fastapi/dev
terminal source .venv/bin/activate
open *.ipynb


