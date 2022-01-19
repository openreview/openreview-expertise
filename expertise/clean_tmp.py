import glob, shutil, tempfile, os, json, logging, sys
from expertise.service.expertise import JobStatus
from openreview import OpenReviewException

logging.basicConfig(encoding='utf-8', level=logging.INFO)

def get_oldest_time(job_dir):
    """Finds the creation date of the oldest running job, returns sys.maxsize if all jobs finished/have an error"""
    oldest_time = sys.maxsize
    root_dir = job_dir

    if not os.path.isdir(root_dir):
        raise OpenReviewException('Error: server has not recieved any jobs/jobs folder is missing')

    subdirs = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
    if len(subdirs) == 0:
        raise OpenReviewException('Error: no jobs in the jobs folder')

    # For each job, read the config cdate and compare
    for dir in subdirs:
        config_dir = os.path.join(root_dir, dir, 'config.json')
        with open(config_dir, 'r') as f:
            config = json.load(f)
        
        # Only consider jobs that are taking up space in the queue
        if config['status'] == JobStatus.ERROR or config['status'] == JobStatus.COMPLETED:
            continue

        if oldest_time == 0:
            oldest_time = config['cdate']
        else:
            oldest_time = min(oldest_time, config['cdate'])

    return oldest_time

def clean_tmp_files(job_dir):
    """Removes files from the tmp directory that are created by SPECTER"""
    logging.info('Cleaning temp files...')
    dirs = glob.glob(f"{tempfile.gettempdir()}/tmp*/")

    del_time = get_oldest_time(job_dir)

    for dir in dirs:
        cdate = int(os.stat(dir).st_ctime * 1000)
        if cdate < del_time:
            logging.info(f"Removing {dir}")
            shutil.rmtree(dir)
        else:
            logging.info(f"Keeping {dir} - Files may be in use")

if __name__ == '__main__':
    clean_tmp_files('jobs')