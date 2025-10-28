1. mkdir hw5
2. cd hw5
3. pip install uv
4. `uv --version` to answer question 1 
5. uv init
6. `uv add scikit-learn==1.6.1`
7. open uv.lock to answer question 2
8. wget https://github.com/DataTalksClub/machine-learning-zoomcamp/raw/refs/heads/master/cohorts/2025/05-deployment/pipeline_v1.bin
9. write function `predict_single` in main.py and `uv run python main.py` 
10. `uv add fastapi uvicorn requests`
11. write `serve.py`
12. `uv run uvicorn serve:app --reload` to launch a predict serve.
13. write `question4.py`
14. open a new terminal and  `cd hw5` , `uv run python question4.py` 
15. `docker pull agrigorev/zoomcamp-model:2025` and `docker images agrigorev/zoomcamp-model:2025`
16. write `Dockerfile` and build image `docker build -t my-zoomcamp-fastapi:latest .` and run `docker run -p 8000:8000 my-zoomcamp-fastapi:latest`
17. run question4.py again