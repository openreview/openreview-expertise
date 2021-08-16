import logging, json, os, shutil
from unittest.mock import MagicMock
from expertise.execute_expertise import *
from expertise.service.server import celery_app as celery

def mock_client():
    client = MagicMock(openreview.Client)

    def get_group(group_id):
        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        group = openreview.Group.from_json(data['groups'][group_id])
        return group

    def search_profiles(confirmedEmails=None, ids=None, term=None):
        with open('tests/data/fakeData.json') as json_file:
            data = json.load(json_file)
        profiles = data['profiles']
        profiles_dict_emails = {}
        profiles_dict_tilde = {}
        for profile in profiles:
            profile = openreview.Profile.from_json(profile)
            if profile.content.get('emails') and len(profile.content.get('emails')):
                profiles_dict_emails[profile.content['emails'][0]] = profile
            profiles_dict_tilde[profile.id] = profile
        if confirmedEmails:
            return_value = {}
            for email in confirmedEmails:
                if profiles_dict_emails.get(email, False):
                    return_value[email] = profiles_dict_emails[email]

        if ids:
            return_value = []
            for tilde_id in ids:
                return_value.append(profiles_dict_tilde[tilde_id])
        return return_value

    client.get_group = MagicMock(side_effect=get_group)
    client.search_profiles = MagicMock(side_effect=search_profiles)

    return client

@celery.task(name='userpaper', track_started=True, bind=True, time_limit=3600 * 24)
def run_userpaper(self, config: dict, logger: logging.Logger, in_test: bool = False):
    try:
        if not in_test:
            openreview_client = openreview.Client(
                token=config['token'],
                baseurl=config['baseurl']
            )
        else:
            openreview_client = mock_client()
            logger.info('Creating dataset')
        execute_create_dataset(openreview_client, config_file=config)
        run_expertise.apply_async(
                (config, logger),
                queue='expertise',
        )
    except Exception as exc:
        working_dir = os.path.join(config['profile_dir'], str(config['job_id']))
        logger.error(f'Removing dir: {working_dir}')
        shutil.rmtree(working_dir)
        logger.error(f"Writing to: {os.path.join(config['profile_dir'], 'err.log')}")
        with open(os.path.join(config['profile_dir'], 'err.log'), 'a+') as f:
            f.write(f"{config['job_id']},")
        logger.error('Error: {}'.format(exc))

@celery.task(name='expertise', track_started=True, bind=True, time_limit=3600 * 24)
def run_expertise(self, config: dict, logger: logging.Logger):
    try:
        logger.info('Executing expertise')
        execute_expertise(config_file=config)
    except Exception as exc:
        working_dir = os.path.join(config['profile_dir'], str(config['job_id']))
        logger.error(f'Removing dir: {working_dir}')
        shutil.rmtree(working_dir)
        logger.error(f"Writing to: {os.path.join(config['profile_dir'], 'err.log')}")
        with open(os.path.join(config['profile_dir'], 'err.log'), 'a+') as f:
            f.write(f"{config['job_id']},")
        logger.error('Error: {}'.format(exc))

