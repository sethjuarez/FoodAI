az ml model deploy --name foodai --model foodai:3 ^
--entry-script score.py --runtime python --conda-file foodai_scoring.yml ^
--deploy-config-file deployconfig.json --compute-target sauron ^
--overwrite