az ml model deploy --name foodai-local --model foodai:8 ^
--entry-script score.py --runtime python --conda-file foodai_scoring.yml ^
--compute-type local --port 32267 ^
-w hal -g robots ^
--overwrite