import os
import subprocess

from .logger import Logger

logger = Logger()

DEFAULT_LOG_DIR = "./logs"

class Utils:

    def __init__(self):
         self.logger = logger
         pass

    def write_log(filepath=DEFAULT_LOG_DIR, data={}):
        return

    def check_if_in_docker_container(self):
        if os.environ.get('APP_ENV') == 'docker':
            return True
        else:
            return False

    def check_for_recent_filechanges(self, interval:float=.05):
        interval = str(interval)
        cmd = ['find', '~/', '-mtime', '-1', '-ls']
        cmd = ' '.join(cmd)
        res = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stderr, stdout = res.communicate()
        print(stderr, stdout)
        return

    def call_recentmost(self, threshold:int=2):
        if self.check_if_in_docker_container():
            cmd = ['find', '~/', '-type', 'f|./app/components/utils/recentmost', '20']
        else:
            cmd = ['find', '~/', '-type', 'f|./components/utils/recentmost', '20']
        cmd = ' '.join(cmd)
        res = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stderr, stdout = res.communicate()
        print(stderr, stdout)
        return
