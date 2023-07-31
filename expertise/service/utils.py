import openreview
import shortuuid
import os
import time
import json
import re
import redis, pickle
from unittest.mock import MagicMock
from enum import Enum

import re
SUPERUSER_IDS = ['openreview.net', 'OpenReview.net', '~Super_User1']

def get_user_id(openreview_client):
    """
    Returns the user id from an OpenReview client for authenticating access

    :param openreview_client: A logged in client with the user credentials
    :type openreview_client: openreview.Client

    :returns id: The id of the logged in user
    """
    user = openreview_client.user
    return user.get('user', {}).get('id') if user else None

def _get_required_field(req, superkey, key):
    try:
        field = req.pop(key)
    except KeyError:
        raise openreview.OpenReviewException(f"Bad request: required field missing in {superkey}: {key}")
    return field

class JobStatus(str, Enum):
    INITIALIZED = 'Initialized'
    QUEUED = 'Queued'
    FETCHING_DATA  = 'Fetching Data'
    EXPERTISE_QUEUED = 'Queued for Expertise'
    RUN_EXPERTISE = 'Running Expertise'
    COMPLETED = 'Completed'
    ERROR = 'Error'

class JobDescription(dict, Enum):
    VALS = {
        JobStatus.INITIALIZED: 'Server received config and allocated space',
        JobStatus.QUEUED: 'Job is waiting to start fetching OpenReview data',
        JobStatus.FETCHING_DATA: 'Job is currently fetching data from OpenReview',
        JobStatus.EXPERTISE_QUEUED: 'Job has assembled the data and is waiting in queue for the expertise model',
        JobStatus.RUN_EXPERTISE: 'Job is running the selected expertise model to compute scores',
        JobStatus.COMPLETED: 'Job is complete and the computed scores are ready',
    }
class APIRequest(object):
    """
    Validates and load objects and fields from POST requests
    """
    def __init__(self, request):
            
        self.entityA = {}
        self.entityB = {}
        self.model = {}
        self.dataset = {}
        root_key = 'request'

        def _get_field_from_request(field):
            return _get_required_field(request, root_key, field)

        def _load_entity_a(entity):
            self._load_entity('entityA', entity, self.entityA)

        def _load_entity_b(entity):
            self._load_entity('entityB', entity, self.entityB)

        # Get the name of the job
        self.name = _get_field_from_request('name')

        # Validate entityA and entityB
        entity_a = _get_field_from_request('entityA')
        entity_b = _get_field_from_request('entityB')

        _load_entity_a(entity_a)
        _load_entity_b(entity_b)

        # Optionally check for model object
        self.model = request.pop('model', {})

        # Optionally check for dataset object
        self.dataset = request.pop('dataset', {})

        # Check for empty request
        if len(request.keys()) > 0:
            raise openreview.OpenReviewException(f"Bad request: unexpected fields in {root_key}: {list(request.keys())}")
    
    def _load_entity(self, entity_id, source_entity, target_entity):
        '''Load information from an entity into the config'''
        def _get_from_entity(key):
            return _get_required_field(source_entity, entity_id, key)

        type = _get_from_entity('type')
        target_entity['type'] = type
        # Handle type group
        if type == 'Group':
            if 'memberOf' in source_entity.keys():
                target_entity['memberOf'] = _get_from_entity('memberOf')
                # Check for optional expertise field
                if 'expertise' in source_entity.keys():
                    target_entity['expertise'] = source_entity.pop('expertise')
            else:
                raise openreview.OpenReviewException(f"Bad request: no valid {type} properties in {entity_id}")
        # Handle type note
        elif type == 'Note':
            if ('invitation' in source_entity.keys() or 'withVenueid' in source_entity.keys()) and 'id' in source_entity.keys():
                raise openreview.OpenReviewException(f"Bad request: only provide a single id or single invitation and/or venue id in {entity_id}")

            if 'invitation' in source_entity.keys():
                target_entity['invitation'] = _get_from_entity('invitation')
            elif 'withVenueid' in source_entity.keys():
                target_entity['withVenueid'] = _get_from_entity('withVenueid')
            elif 'id' in source_entity.keys():
                target_entity['id'] = _get_from_entity('id')
            else:
                raise openreview.OpenReviewException(f"Bad request: no valid {type} properties in {entity_id}")
            
            if 'withContent' in source_entity.keys():
                target_entity['withContent'] = _get_from_entity('withContent')
        else:
            raise openreview.OpenReviewException(f"Bad request: invalid type in {entity_id}")

        # Check for extra entity fields
        if len(source_entity.keys()) > 0:
            raise openreview.OpenReviewException(f"Bad request: unexpected fields in {entity_id}: {list(source_entity.keys())}")
        
    def to_json(self):
        body = {
            'name': self.name,
            'entityA': self.entityA,
            'entityB': self.entityB,
        }

        if len(self.model.keys()) > 0:
            body['model'] = self.model
        if hasattr(self, 'dataset'):
            if len(self.dataset.keys()) > 0:
                body['dataset'] = self.dataset

        return body

class RedisDatabase(object):
    """
    Communicates with the local Redis instance to store and load jobs
    """
    def __init__(self,
        host=None,
        port=None,
        db=None,
        connection_pool=None) -> None:
        if not connection_pool:
            self.db = redis.Redis(
                host = host,
                port = port,
                db = db
            )
        else:
            self.db = redis.Redis(connection_pool=connection_pool)
    def save_job(self, job_config):
        self.db.set(f"job:{job_config.job_id}", pickle.dumps(job_config))
    
    def load_all_jobs(self, user_id):
        """
        Searches all keys for configs with matching user id
        If a Redis entry exists but the files do not, remove the entry from Redis and do not return this job
        Returns empty list if no jobs found
        """
        configs = []

        for job_key in self.db.scan_iter("job:*"):
            current_config = pickle.loads(self.db.get(job_key))

            if not os.path.isdir(current_config.job_dir):
                print(f"No files found {job_key} - skipping")
                continue

            if current_config.user_id == user_id or user_id in SUPERUSER_IDS:
                configs.append(current_config)

        return configs

    def load_job(self, job_id, user_id):
        """
        Retrieves a config based on job id
        """
        job_key = f"job:{job_id}"

        if not self.db.exists(job_key):
            raise openreview.OpenReviewException('Job not found')        
        config = pickle.loads(self.db.get(job_key))
        if not os.path.isdir(config.job_dir):
            self.remove_job(user_id, job_id)
            raise openreview.OpenReviewException('Job not found')

        if config.user_id != user_id and user_id not in SUPERUSER_IDS:
            raise openreview.OpenReviewException('Forbidden: Insufficient permissions to access job')

        return config
    
    def remove_job(self, user_id, job_id):
        job_key = f"job:{job_id}"

        if not self.db.exists(job_key):
            raise openreview.OpenReviewException('Job not found')
        config = pickle.loads(self.db.get(job_key))
        if config.user_id != user_id and user_id not in SUPERUSER_IDS:
            raise openreview.OpenReviewException('Forbidden: Insufficient permissions to modify job')

        self.db.delete(job_key)
        return config

class JobConfig(object):
    """
    Helps translate fields from API requests to fields usable by the expertise system
    """
    def __init__(self,
        name=None,
        user_id=None,
        job_id=None,
        baseurl=None,
        baseurl_v2=None,
        job_dir=None,
        cdate=None,
        mdate=None,
        status=None,
        description=None,
        match_group=None,
        alternate_match_group=None,
        dataset=None,
        model=None,
        exclusion_inv=None,
        inclusion_inv=None,
        alternate_exclusion_inv=None,
        alternate_inclusion_inv=None,
        paper_invitation=None,
        paper_venueid=None,
        paper_content=None,
        paper_id=None,
        model_params=None):
        
        self.name = name
        self.user_id = user_id
        self.job_id = job_id
        self.baseurl = baseurl
        self.baseurl_v2 = baseurl_v2
        self.job_dir = job_dir
        self.cdate = cdate
        self.mdate = mdate
        self.status = status
        self.description = description
        self.match_group = match_group
        self.alternate_match_group = alternate_match_group
        self.dataset = dataset
        self.model = model
        self.exclusion_inv = exclusion_inv
        self.inclusion_inv = inclusion_inv
        self.alternate_exclusion_inv = alternate_exclusion_inv
        self.alternate_inclusion_inv = alternate_inclusion_inv
        self.paper_invitation = paper_invitation
        self.paper_venueid = paper_venueid
        self.paper_content = paper_content
        self.paper_id = paper_id
        self.model_params = model_params

        self.api_request = None

    def to_json(self):
        pre_body = {
            'name': self.name,
            'user_id': self.user_id,
            'job_id': self.job_id,
            'baseurl': self.baseurl,
            'baseurl_v2': self.baseurl_v2,
            'job_dir': self.job_dir,
            'cdate': self.cdate,
            'mdate': self.mdate,
            'match_group': self.match_group,
            'alternate_match_group': self.alternate_match_group,
            'dataset': self.dataset,
            'model': self.model,
            'exclusion_inv': self.exclusion_inv,
            'inclusion_inv': self.inclusion_inv,
            'alternate_exclusion_inv': self.alternate_exclusion_inv,
            'alternate_inclusion_inv': self.alternate_inclusion_inv,
            'paper_invitation': self.paper_invitation,
            'paper_venueid': self.paper_venueid,
            'paper_content': self.paper_content,
            'paper_id': self.paper_id,
            'model_params': self.model_params
        }

        # Remove objects that are none
        body = {}
        body_items = pre_body.items()
        for key, val in body_items:
            # Allow a None token
            if val is not None or key == 'token':
                body[key] = val

        return body

    def from_request(api_request: APIRequest,
        starting_config = {},
        openreview_client = None,
        openreview_client_v2 = None,
        server_config = {},
        working_dir = None):
        """
        Sets default fields from the starting_config and attempts to override from api_request fields
        """
        def _camel_to_snake(camel_str):
            camel_str = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', camel_str).lower()

        descriptions = JobDescription.VALS.value
        config = JobConfig()

        # Set metadata fields from request
        config.name = api_request.name
        config.user_id = get_user_id(openreview_client)
        config.job_id = shortuuid.ShortUUID().random(length=5)
        config.baseurl = server_config['OPENREVIEW_BASEURL']
        config.baseurl_v2 = server_config['OPENREVIEW_BASEURL_V2']
        config.api_request = api_request    

        root_dir = os.path.join(working_dir, config.job_id)
        config.job_dir = root_dir
        config.cdate = int(time.time() * 1000)
        config.mdate = config.cdate
        config.status = JobStatus.INITIALIZED.value
        config.description = descriptions[JobStatus.INITIALIZED]

        # Handle Group cases
        config.match_group = starting_config.get('match_group', None)
        config.alternate_match_group = starting_config.get('alternate_match_group', None)
        config.inclusion_inv = None
        config.exclusion_inv = None
        config.alternate_inclusion_inv = None
        config.alternate_exclusion_inv = None

        # TODO: Need new keyword

        if api_request.entityA['type'] == 'Group':
            config.match_group = [api_request.entityA['memberOf']]
            edge_inv = api_request.entityA.get('expertise', None)

            if edge_inv:
                edge_inv_id = edge_inv.get('exclusion', {}).get('invitation', None)
                if edge_inv_id is None:
                    edge_inv_id = edge_inv.get('invitation', None)
                if edge_inv_id is None or len(edge_inv_id) <= 0:
                    raise openreview.OpenReviewException('Bad request: Expertise invitation indicated but ID not provided')

                try:
                    label = openreview_client.get_invitation(edge_inv_id).reply.get('content', {}).get('label', {}).get('value-radio',['Include'])[0]
                except openreview.OpenReviewException as e:
                    if "notfound" in str(e).lower():
                        label = openreview_client_v2.get_invitation(edge_inv_id).edit.get('label', {}).get('param', {}).get('enum',['Include'])[0]
                    else:
                        raise e

                if 'exclude' not in label.lower():
                    config.inclusion_inv = edge_inv_id
                else:
                    config.exclusion_inv = edge_inv_id

        if api_request.entityB['type'] == 'Group':
            config.alternate_match_group = [api_request.entityB['memberOf']]
            edge_inv = api_request.entityB.get('expertise', None)

            if edge_inv:
                edge_inv_id = edge_inv.get('exclusion', {}).get('invitation', None)
                if edge_inv_id is None:
                    edge_inv_id = edge_inv.get('invitation', None)
                if edge_inv_id is None:
                    raise openreview.OpenReviewException('Bad request: Expertise invitation indicated but ID not provided')

                try:
                    label = openreview_client.get_invitation(edge_inv_id).reply.get('content', {}).get('label', {}).get('value-radio',['Include'])[0]
                except openreview.OpenReviewException as e:
                    if "notfound" in str(e).lower():
                        label = openreview_client_v2.get_invitation(edge_inv_id).edit.get('label', {}).get('param', {}).get('enum',['Include'])[0]
                    else:
                        raise e

                if 'include' in label.lower():
                    config.alternate_inclusion_inv = edge_inv_id
                else:
                    config.alternate_exclusion_inv = edge_inv_id

        # Handle Note cases
        config.paper_invitation = None
        config.paper_id = None

        if api_request.entityA['type'] == 'Note':
            inv, id, venueid, content = api_request.entityA.get('invitation', None), api_request.entityA.get('id', None), api_request.entityA.get('withVenueid', None), api_request.entityA.get('withContent', None)

            if inv:
                config.paper_invitation = inv
            if id:
                config.paper_id = id
            if venueid:
                config.paper_venueid = venueid
            if content:
                config.paper_content = content

        elif api_request.entityB['type'] == 'Note':
            inv, id, venueid, content = api_request.entityB.get('invitation', None), api_request.entityB.get('id', None), api_request.entityB.get('withVenueid', None), api_request.entityB.get('withContent', None)

            if inv:
                config.paper_invitation = inv
            if id:
                config.paper_id = id
            if venueid:
                config.paper_venueid = venueid
            if content:
                config.paper_content = content                

        # Validate that other paper fields are none if an alternate match group is present
        if config.alternate_match_group is not None and (config.paper_id is not None or config.paper_invitation is not None):
            raise openreview.OpenReviewException('Bad request: Cannot provide paper id/invitation and alternate match group')

        # Load optional dataset params from default config
        allowed_dataset_params = [
            'minimumPubDate',
            'topRecentPubs'
        ]
        config.dataset = starting_config.get('dataset', {})
        config.dataset['directory'] = root_dir

        # Attempt to load any API request dataset params
        dataset_params = api_request.dataset
        if dataset_params:
            for param in dataset_params.keys():
                # Handle general case
                if param not in allowed_dataset_params:
                    raise openreview.OpenReviewException(f"Bad request: unexpected fields in model: {[param]}")

                snake_param = _camel_to_snake(param)
                config.dataset[snake_param] = dataset_params[param]

        # Load optional model params from default config
        path_fields = ['work_dir', 'scores_path', 'publications_path', 'submissions_path']
        allowed_model_params = [
            'name',
            'sparseValue',
            'useTitle',
            'useAbstract',
            'scoreComputation',
            'skipSpecter'
        ]
        config.model = starting_config.get('model', None)
        model_params = starting_config.get('model_params', {})
        config.model_params = {}
        config.model_params['use_title'] = model_params.get('use_title', None)
        config.model_params['use_abstract'] = model_params.get('use_abstract', None)
        config.model_params['average_score'] = model_params.get('average_score', None)
        config.model_params['max_score'] = model_params.get('max_score', None)
        config.model_params['skip_specter'] = model_params.get('skip_specter', None)
        config.model_params['specter_batch_size'] = model_params.get('specter_batch_size', 16)
        config.model_params['mfr_batch_size'] = model_params.get('mfr_batch_size', 50)
        config.model_params['sparse_value'] = model_params.get('sparse_value', 300)
        config.model_params['use_cuda'] = model_params.get('use_cuda', False)
        config.model_params['use_redis'] = model_params.get('use_redis', False)

        # Attempt to load any API request model params
        api_model = api_request.model
        if api_model:
            for param in api_model.keys():
                # Handle special cases
                if param == 'scoreComputation':
                    compute_with = api_model.get('scoreComputation', None)
                    if compute_with == 'max':
                        config.model_params['max_score'] = True
                        config.model_params['average_score'] = False
                    elif compute_with == 'avg':
                        config.model_params['max_score'] = False
                        config.model_params['average_score'] = True
                    else:
                        raise openreview.OpenReviewException("Bad request: invalid value in field 'scoreComputation' in 'model' object")
                    continue

                if param == 'name': ## BUG
                    config.model = api_model.get('name', config.model)
                
                # Handle general case
                if param not in allowed_model_params:
                    raise openreview.OpenReviewException(f"Bad request: unexpected fields in model: {[param]}")

                snake_param = _camel_to_snake(param)
                config.model_params[snake_param] = api_model[param]
        
        # Set server-side path fields
        for field in path_fields:
            config.model_params[field] = root_dir

        if 'specter' in config.model:
            config.model_params['specter_dir'] = server_config['SPECTER_DIR']
        if 'mfr' in config.model:
            config.model_params['mfr_feature_vocab_file'] = server_config['MFR_VOCAB_DIR']
            config.model_params['mfr_checkpoint_dir'] = server_config['MFR_CHECKPOINT_DIR']

        return config
    
    def from_json(job_config):
        config = JobConfig(
            name = job_config.get('name'),
            user_id = job_config.get('user_id'),
            job_id = job_config.get('job_id'),
            baseurl = job_config.get('baseurl'),
            baseurl_v2 = job_config.get('baseurl_v2'),
            job_dir = job_config.get('job_dir'),
            cdate = job_config.get('cdate'),
            mdate = job_config.get('mdate'),
            status = job_config.get('status'),
            description = job_config.get('description'),
            match_group = job_config.get('match_group'),
            alternate_match_group=job_config.get('alternate_match_group'),
            dataset = job_config.get('dataset'),
            model = job_config.get('model'),
            exclusion_inv = job_config.get('exclusion_inv'),
            inclusion_inv = job_config.get('inclusion_inv'),
            alternate_exclusion_inv = job_config.get('alternate_exclusion_inv'),
            alternate_inclusion_inv = job_config.get('alternate_inclusion_inv'),
            paper_invitation = job_config.get('paper_invitation'),
            paper_venueid = job_config.get('paper_venueid'),
            paper_content = job_config.get('paper_content'),
            paper_id = job_config.get('paper_id'),
            model_params = job_config.get('model_params')
        )
        return config
