import logging, sys, runpy
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True, format='%(message)s')
runpy.run_path('/root/autodl-tmp/RoboSynChallenge/policy/act/scripts/train.py', run_name='__main__')
