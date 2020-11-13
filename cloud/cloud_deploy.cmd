az ml model deploy --name foodai --model foodai:11 ^
--entry-script ../src/score.py --runtime python --conda-file foodai_scoring.yml ^
--deploy-config-file deployconfig.json --compute-target sauron --ai True ^
--overwrite