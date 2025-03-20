import openreview
import shortuuid
import os
import time
import json
import re
import datetime
import redis, pickle
import logging
from unittest.mock import MagicMock
from enum import Enum
import google.cloud.aiplatform as aip
from google.cloud import storage
from google.cloud.aiplatform_v1.types import PipelineState

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
        JobStatus.ERROR: 'Job has encountered an error and has failed to complete',
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
            elif 'reviewerIds' in source_entity.keys():
                target_entity['reviewerIds'] = _get_from_entity('reviewerIds')
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

            if 'submissionIds' in source_entity.keys():
                target_entity['submissionIds'] = _get_from_entity('submissionIds')
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
        connection_pool=None,
        sync_on_disk=True) -> None:
        if not connection_pool:
            self.db = redis.Redis(
                host = host,
                port = port,
                db = db
            )
        else:
            self.db = redis.Redis(connection_pool=connection_pool)

        self.sync_on_disk = sync_on_disk
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

            if self.sync_on_disk and not os.path.isdir(current_config.job_dir):
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
        if self.sync_on_disk and not os.path.isdir(config.job_dir):
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
        cloud_id=None,
        baseurl=None,
        baseurl_v2=None,
        job_dir=None,
        cdate=None,
        mdate=None,
        status=None,
        description=None,
        match_group=None,
        match_paper_invitation=None,
        match_paper_venueid=None,
        match_paper_id=None,
        match_paper_content=None,
        alternate_match_group=None,
        reviewer_ids=None,
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
        self.cloud_id = cloud_id
        self.baseurl = baseurl
        self.baseurl_v2 = baseurl_v2
        self.job_dir = job_dir
        self.cdate = cdate
        self.mdate = mdate
        self.status = status
        self.description = description
        self.match_group = match_group
        self.match_paper_invitation = match_paper_invitation
        self.match_paper_venueid = match_paper_venueid
        self.match_paper_id = match_paper_id
        self.match_paper_content = match_paper_content
        self.alternate_match_group = alternate_match_group
        self.reviewer_ids = reviewer_ids
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
        json_keys = [
            'name',
            'user_id',
            'job_id',
            'cloud_id',
            'baseurl',
            'baseurl_v2',
            'job_dir',
            'cdate',
            'mdate',
            'status',
            'description',
            'match_group',
            'match_paper_invitation',
            'match_paper_venueid',
            'match_paper_id',
            'match_paper_content',
            'alternate_match_group',
            'reviewer_ids',
            'dataset',
            'model',
            'exclusion_inv',
            'inclusion_inv',
            'alternate_exclusion_inv',
            'alternate_inclusion_inv',
            'paper_invitation',
            'paper_venueid',
            'paper_content',
            'paper_id',
            'model_params'
        ]


        # Build the JSON dictionary using getattr to fetch values.
        body = {}
        for key in json_keys:
            value = getattr(self, key, None)
            if value is not None:
                body[key] = value

        return body

    def from_request(api_request: APIRequest,
        job_id=None,
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

        def _populate_note_fields(entity, config, paper_paper_scoring=False):
            inv, id, venueid, content = entity.get('invitation', None), entity.get('id', None), entity.get('withVenueid', None), entity.get('withContent', None)

            if paper_paper_scoring:
                if inv:
                    config.match_paper_invitation = inv
                if id:
                    config.match_paper_id = id
                if venueid:
                    config.match_paper_venueid = venueid
                if content:
                    config.match_paper_content = content
            else:
                if inv:
                    config.paper_invitation = inv
                if id:
                    config.paper_id = id
                if venueid:
                    config.paper_venueid = venueid
                if content:
                    config.paper_content = content

        descriptions = JobDescription.VALS.value
        config = JobConfig()

        # Set metadata fields from request
        config.name = api_request.name
        config.user_id = get_user_id(openreview_client)
        config.job_id = shortuuid.ShortUUID().random(length=5) if job_id is None else job_id
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
            if 'memberOf' in api_request.entityA:
                config.match_group = [api_request.entityA['memberOf']]
            elif 'reviewerIds' in api_request.entityA:
                config.reviewer_ids = api_request.entityA['reviewerIds']
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
            _populate_note_fields(api_request.entityA, config)

        if api_request.entityB['type'] == 'Note':
            if api_request.entityA['type'] == 'Note':
                _populate_note_fields(api_request.entityB, config, paper_paper_scoring=True)
            else:
                _populate_note_fields(api_request.entityB, config)

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
            'skipSpecter',
            'useCuda'
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

                if param == 'name':
                    config.model = api_model.get('name', config.model)
                
                # Handle general case
                if param not in allowed_model_params:
                    raise openreview.OpenReviewException(f"Bad request: unexpected fields in model: {[param]}")

                snake_param = _camel_to_snake(param)
                config.model_params[snake_param] = api_model[param]
        
        # Set server-side path fields
        for field in path_fields:
            config.model_params[field] = root_dir

        # Infer compute_paper_paper from two Note entities
        valid_paper_paper_models = ['specter', 'specter2', 'scincl', 'specter2+scincl']
        config.model_params['compute_paper_paper'] = False
        if api_request.entityA['type'] == 'Note' and api_request.entityB['type'] == 'Note':
            if config.model not in valid_paper_paper_models:
                raise openreview.OpenReviewException(f"Bad request: model {config.model} does not support paper-paper scoring")
            config.model_params['compute_paper_paper'] = True

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
            match_paper_invitation = job_config.get('match_paper_invitation'),
            match_paper_venueid = job_config.get('match_paper_venueid'),
            match_paper_id = job_config.get('match_paper_id'),
            alternate_match_group=job_config.get('alternate_match_group'),
            reviewer_ids=job_config.get('reviewer_ids'),
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

class GCPInterface(object):
    """
    Provides an interface to GCP for requesting, monitoring and managing expertise jobs
    """

    GCS_STATE_TO_JOB_STATE = {
        PipelineState.PIPELINE_STATE_PENDING: JobStatus.INITIALIZED,
        PipelineState.PIPELINE_STATE_QUEUED: JobStatus.QUEUED,
        PipelineState.PIPELINE_STATE_RUNNING: JobStatus.RUN_EXPERTISE,
        PipelineState.PIPELINE_STATE_SUCCEEDED: JobStatus.COMPLETED,
        PipelineState.PIPELINE_STATE_FAILED: JobStatus.ERROR,
    }

    def __init__(
        self,
        config=None,
        project_id=None,
        project_number=None,
        region=None,
        pipeline_root=None,
        pipeline_name=None,
        pipeline_repo=None,
        bucket_name=None, 
        jobs_folder=None,
        service_label=None,
        openreview_client=None,
        pipeline_tag='latest',
        logger=None,
        gcs_client=None
    ):

        if config is not None:
            self.project_id = config['GCP_PROJECT_ID']
            self.project_number = config['GCP_PROJECT_NUMBER']
            self.region = config['GCP_REGION']
            self.pipeline_root = config['GCP_PIPELINE_ROOT']
            self.pipeline_name = config['GCP_PIPELINE_NAME']
            self.pipeline_repo = config['GCP_PIPELINE_REPO']
            self.pipeline_tag = config['GCP_PIPELINE_TAG']
            self.bucket_name = config['GCP_BUCKET_NAME']
            self.jobs_folder = config['GCP_JOBS_FOLDER']
            self.service_label = config['GCP_SERVICE_LABEL']
        else:
            self.project_id = project_id
            self.project_number = project_number
            self.region = region
            self.pipeline_root = pipeline_root
            self.pipeline_name = pipeline_name
            self.pipeline_repo = pipeline_repo
            self.pipeline_tag = pipeline_tag
            self.bucket_name = bucket_name
            self.jobs_folder = jobs_folder
            self.service_label = service_label

        required_fields = [
            self.project_id,
            self.project_number,
            self.region,
            self.pipeline_root,
            self.pipeline_name,
            self.pipeline_repo,
            self.pipeline_tag,
            self.bucket_name,
            self.jobs_folder,
            self.service_label
        ]
        
        self.client = openreview_client
        self.request_fname = "request.json"
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        self.logger = logger
        
        if not any(field is None for field in required_fields):
            # Only init AIP if all fields are present to access the project
            self.logger.info(f"Init AIPlatform with project {self.project_id} and region {self.region}")
            aip.init(
                project=project_id,
                location=region
            )

        self.logger.info(f"Init GCS client with project {self.project_id}")
        self.gcs_client = gcs_client or storage.Client(
            project=project_id
        )
        self.logger.info(f"Get bucket {self.bucket_name}")
        self.bucket = self.gcs_client.bucket(self.bucket_name)

    def set_client(self, client):
        self.client = client

    def _generate_vertex_prefix(api_request):
        group_entity = None
        if api_request.entityA['type'] == 'Group':
            group_entity = api_request.entityA
        elif api_request.entityB['type'] == 'Group':
            group_entity = api_request.entityB

        # Handle group-group request
        if api_request.entityA['type'] == 'Group' and api_request.entityB['type'] == 'Group':
            return f"group-{api_request.entityA['memberOf']}"

        # Handle X-note requests
        match_note_entity, note_entity = None, None
        if api_request.entityA['type'] == 'Note':
            note_entity = api_request.entityA
        if api_request.entityB['type'] == 'Note':
            if note_entity is None:
                note_entity = api_request.entityB
            else:
                match_note_entity = api_request.entityB

        if note_entity is None:
            raise openreview.OpenReviewException('Bad request: No note entity found')
        
        # Handle note-note request
        if match_note_entity is not None:
            note_fields = ['invitation', 'withVenueid', 'id']
            match_prefix, submission_prefix = None, None
            for field in note_fields:
                if field in match_note_entity:
                    match_prefix = match_note_entity[field]
                if field in note_entity:
                    submission_prefix = note_entity[field]

            if match_prefix is None or submission_prefix is None:
                raise openreview.OpenReviewException('Bad request: No match or submission prefix found')

            return f"{match_prefix}-{submission_prefix}"
                    
        # Handle group-invitation request
        if 'invitation' in note_entity:
            return f"inv-{group_entity['memberOf']}"
        # Handle group-withVenueid request
        elif 'withVenueid' in note_entity:
            return f"venueid-{group_entity['memberOf']}"
        # Handle group-noteId request
        elif 'id' in note_entity:
            return f"pid-{note_entity['id']}-{group_entity['memberOf']}"

        # Handle group-invitation request
        if 'invitation' in note_entity:
            return f"inv-{group_entity['memberOf']}"
        # Handle group-withVenueid request
        elif 'withVenueid' in note_entity:
            return f"venueid-{group_entity['memberOf']}"
        # Handle group-noteId request
        elif 'id' in note_entity:
            return f"pid-{note_entity['id']}-{group_entity['memberOf']}"

    def create_job(self, json_request: dict, user_id: str = None, client = None):
        def create_folder(bucket_name, folder_path):
            client = storage.Client()
            bucket = client.get_bucket(bucket_name)
            blob = bucket.blob(f"{folder_path}/")
            blob.upload_from_string('')
            self.logger.info(f"Folder '{folder_path}' created in bucket '{bucket_name}'.")

        def create_folder_if_not_exists(bucket_name, folder_path):
            client = storage.Client()
            bucket = client.bucket(bucket_name)

            # Check if the folder exists by listing blobs with a prefix
            blobs = list(bucket.list_blobs(prefix=f"{folder_path}/", max_results=1))
            if not blobs:
                # If the folder doesn't exist, create a "dummy" blob to simulate the folder
                blob = bucket.blob(f"{folder_path}/")
                blob.upload_from_string('')
                self.logger.info(f"Folder '{folder_path}' created in bucket '{bucket_name}'.")

        def write_json_to_gcs(bucket_name, folder_path, file_name, data):
            create_folder_if_not_exists(bucket_name, folder_path)
            client = storage.Client()
            bucket = client.bucket(bucket_name)

            blob = bucket.blob(f"{folder_path}/{file_name}")

            blob.upload_from_string(
                data=json.dumps(data),
                content_type="application/json"
            )
            self.logger.info(f"JSON file '{file_name}' written to '{folder_path}' in bucket '{bucket_name}'.")

        or_client = client if client else self.client
        api_request = APIRequest(json_request)
        job_id = GCPInterface._generate_vertex_prefix(api_request) + '-' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        valid_vertex_id = job_id.replace('/','-').replace(':','-').replace('_','-').replace('.', '-').lower()

        folder_path = f"{self.jobs_folder}/{valid_vertex_id}"
        data = api_request.to_json()

        # Expected fields
        data['name'] = valid_vertex_id

        # Popped fields
        data['token'] = or_client.token
        data['baseurl_v1'] = openreview.tools.get_base_urls(or_client)[0]
        data['baseurl_v2'] = openreview.tools.get_base_urls(or_client)[1]
        data['gcs_folder'] = f"gs://{self.bucket_name}/{folder_path}"
        #data['dump_embs'] = True
        #data['dump_archives'] = True

        # Deleted metadata fields before hitting the pipeline
        data['user_id'] = user_id if user_id else get_user_id(or_client)
        data['cdate'] = int(time.time() * 1000)

        write_json_to_gcs(self.bucket_name, folder_path, self.request_fname, data)

        job = aip.PipelineJob(
            display_name = valid_vertex_id,
            template_path = f"https://{self.region}-kfp.pkg.dev/{self.project_id}/{self.pipeline_repo}/{self.pipeline_name}/{self.pipeline_tag}",
            job_id = valid_vertex_id,
            pipeline_root = f"gs://{self.bucket_name}/{self.pipeline_root}",
            parameter_values = {'job_config': json.dumps(data)},
            labels = self.service_label)

        job.submit()

        return valid_vertex_id

    def get_job_status_by_job_id(self, user_id, job_id):
        job_blobs = self.bucket.list_blobs(prefix=f"{self.jobs_folder}/{job_id}")
        self.logger.info(f"Searching for job {job_id} | prefix={self.jobs_folder}/{job_id}")
        all_requests = [
            json.loads(blob.download_as_string()) for blob in job_blobs if self.request_fname in blob.name
        ]
        authenticated_requests = [
            req for req in all_requests if user_id == req['user_id'] or user_id in SUPERUSER_IDS
        ]
        if len(all_requests) == 0:
            raise openreview.OpenReviewException('Job not found')
        if len(authenticated_requests) == 0:
            raise openreview.OpenReviewException('Forbidden: Insufficient permissions to access job')
        if len(authenticated_requests) > 1:
            raise openreview.OpenReviewException('Internal Error: Multiple requests found for job')

        request = authenticated_requests[0]
        job = aip.PipelineJob.get(f"projects/{self.project_number}/locations/{self.region}/pipelineJobs/{job_id}")

        descriptions = JobDescription.VALS.value
        status = GCPInterface.GCS_STATE_TO_JOB_STATE.get(job.state, '')
        description = descriptions[status]

        return {
                'name': job_id,
                'tauthor': user_id,
                'jobId': job_id,
                'status': status,
                'description': description,
                'cdate': request['cdate'],
                'mdate': int(job.update_time.timestamp() * 1000),
                'request': request
            }

    def get_job_status(
        self,
        user_id,
        query_params,
    ):
        # search bucket
        def check_status(job):
            status = GCPInterface.GCS_STATE_TO_JOB_STATE.get(job.state, '')
            search_status = query_obj.get('status', '')
            return not search_status or status.lower().startswith(search_status.lower())
        
        def check_member(request):
            search_member, memberOf = '', ''
            if 'memberOf' in query_obj.keys():
                memberOf = request.get('entityA', {}).get('memberOf', '') or request.get('entityB', {}).get('memberOf', '')
                search_member = query_obj['memberOf']

            elif 'memberOf' in query_obj.get('entityA', {}).keys():
                memberOf = request.get('entityA', {}).get('memberOf', '')
                search_member = query_obj['entityA']['memberOf']

            elif 'memberOf' in query_obj.get('entityB', {}).keys():
                memberOf = request.get('entityB', {}).get('memberOf', '')
                search_member = query_obj['entityB']['memberOf']
            
            return not search_member or memberOf.lower().startswith(search_member.lower())
        
        def check_invitation(request):
            search_invitation, inv = '', ''
            if 'invitation' in query_obj.keys():
                inv = request.get('entityA', {}).get('invitation', '') or request.get('entityB', {}).get('invitation', '')
                search_invitation = query_obj['invitation']

            elif 'invitation' in query_obj.get('entityA', {}).keys():
                inv = request.get('entityA', {}).get('invitation', '')
                search_invitation = query_obj['entityA']['invitation']

            elif 'invitation' in query_obj.get('entityB', {}).keys():
                inv = request.get('entityB', {}).get('invitation', '')
                search_invitation = query_obj['entityB']['invitation']

            return not search_invitation or inv.lower().startswith(search_invitation.lower())

        def check_paper_id(request):
            search_paper_id, paper_id = '', ''
            if 'id' in query_obj.keys():
                paper_id = request.get('entityA', {}).get('id', '') or request.get('entityB', {}).get('id', '')
                search_paper_id = query_obj['id']

            elif 'id' in query_obj.get('entityA', {}).keys():
                paper_id = request.get('entityA', {}).get('id', '')
                search_paper_id = query_obj['entityA']['id']

            elif 'id' in query_obj.get('entityB', {}).keys():
                paper_id = request.get('entityB', {}).get('id', '')
                search_paper_id = query_obj['entityB']['id']

            return not search_paper_id or paper_id.lower().startswith(search_paper_id.lower())

        def check_all_except_status(request):
            return False not in [
                check_member(request),
                check_invitation(request),
                check_paper_id(request)
            ]

        def check_result(request, job):
            return False not in [
                check_status(job),
                check_member(request),
                check_invitation(request),
                check_paper_id(request)
            ]

        def sanitize(name):
            return name.replace('/', '-').replace(':', '-').replace('_', '-').replace('.', '-').lower()

        def create_bucket_prefixes(params):
            paper_id =  params.get('entityA', {}).get('id', '') or params.get('entityB', {}).get('id', '') or params.get('id', '')
            group_id = params.get('entityA', {}).get('memberOf', '') or params.get('entityB', {}).get('memberOf', '') or params.get('memberOf', '')
            inv = params.get('entityA', {}).get('invitation', '') or params.get('entityB', {}).get('invitation', '') or params.get('invitation', '')
            
            base_prefix = f"{self.jobs_folder}/"
            
            if paper_id:
                raw_prefix = f"pid-{paper_id}-{group_id}"
                sanitized = sanitize(raw_prefix)
                return f"{base_prefix}{sanitized}"
            elif group_id:
                if inv:
                    raw_prefix = f"inv-{group_id}"
                    sanitized = sanitize(raw_prefix)
                    return f"{base_prefix}{sanitized}"
                else:
                    raw_prefix = f"venueid-{group_id}"
                    sanitized = sanitize(raw_prefix)
                    return f"{base_prefix}{sanitized}"
            return base_prefix


        result = {'results': []}
        query_obj = {}
        '''
        {
            'paperId': value,
            'entityA': {
                'id': value
            }
        }
        '''

        for query, value in query_params.items():
            if query.find('.') < 0: ## If no entity, store value
                query_obj[query] = value
            else:
                entity, query_by = query.split('.') ## If entity, store value in entity obj
                if entity not in query_obj.keys():
                    query_obj[entity] = {}
                query_obj[entity][query_by] = value
        self.logger.info(f"Query object: {query_obj}")

        all_requests = []
        prefix = create_bucket_prefixes(query_obj)
        for blob in self.bucket.list_blobs(prefix=prefix):
            if self.request_fname in blob.name:
                all_requests.append(json.loads(blob.download_as_string()))

        authenticated_requests = [
            req for req in all_requests if user_id == req['user_id'] or user_id in SUPERUSER_IDS
        ]
        ## Shortlist by all but status
        shortlist = []
        for request in authenticated_requests:
            if check_all_except_status(request):
                shortlist.append(request)

        # If none shortlisted, search all requests anyway
        if len(shortlist) == 0:
            shortlist = authenticated_requests

        for request in shortlist:
            request_name = request['name']
            try:
                job = aip.PipelineJob.get(f"projects/{self.project_number}/locations/{self.region}/pipelineJobs/{request_name}")
            except Exception as e:
                if '404' in str(e):
                    self.logger.info(f"No pipeline for job {request_name}")
                    continue
                else:
                    raise e

            descriptions = JobDescription.VALS.value
            status = GCPInterface.GCS_STATE_TO_JOB_STATE.get(job.state, '')
            description = descriptions[status]

            if check_result(request, job):
                result['results'].append(
                    {
                        'name': request_name,
                        'tauthor': user_id,
                        'jobId': request_name,
                        'status': status,
                        'description': description,
                        'cdate': request['cdate'],
                        'mdate': int(job.update_time.timestamp() * 1000),
                        'request': request
                    }
                )
        return result

    def get_job_results(self, user_id, job_id, delete_on_get=False):

        def _get_scores_and_metadata(all_blobs, job_id, group_scoring=False, paper_scoring=False):
            """
            Extracts the scores and metadata from the GCS bucket

            :param all_blobs: A list of all blobs for a given job
            :type all_blobs: list
            :param job_id: Unique job ID
            :type job_id: str
            :param group_scoring: Indicator for scores between groups
            :type group_scoring: bool
            :param paper_scoring: Indicator for scores between papers
            :type paper_scoring: bool

            :returns scores: The scores as a list of JSONs
            :returns metadata: The metadata as a dictionary
            """
            metadata_files = [
                blob for blob in all_blobs if 'metadata.json' in blob.name
            ]
            score_files = [
                blob for blob in all_blobs if '.jsonl' in blob.name and job_id in blob.name
            ]
            skip_sparse = group_scoring or paper_scoring

            if len(metadata_files) != 1:
                raise openreview.OpenReviewException(f"Internal Error: incorrect metadata files found expected [1] found {len(metadata_files)}")
            if len(score_files) < 1 or len(score_files) > 2:
                raise openreview.OpenReviewException(f"Internal Error: incorrect score files found expected [1, 2] found {len(score_files)}")

            if not skip_sparse:
                sparse_score_files = [
                    blob for blob in score_files if 'sparse' in blob.name
                ]
                if len(sparse_score_files) != 1:
                    raise openreview.OpenReviewException(f"Internal Error: incorrect sparse score files found expected [1] found {len(sparse_score_files)}")
                scores_str = sparse_score_files[0].download_as_string()
                if isinstance(scores_str, bytes):
                    scores_str = scores_str.decode('utf-8')
            else:
                non_sparse_score_files = [
                    blob for blob in score_files if 'sparse' not in blob.name
                ]
                if len(non_sparse_score_files) != 1:
                    scoring_type_string = 'group' if group_scoring else 'paper'
                    raise openreview.OpenReviewException(f"Internal Error: incorrect {scoring_type_string} score files found expected [1] found {len(non_sparse_score_files)}")
                scores_str = non_sparse_score_files[0].download_as_string()
                if isinstance(scores_str, bytes):
                    scores_str = scores_str.decode('utf-8')

            metadata = json.loads(metadata_files[0].download_as_string())
            scores = [json.loads(line) for line in scores_str.split('\n') if line != '']

            return {
                'results': scores,
                'metadata': metadata
            }

        # convert to csv
        job_blobs = list(self.bucket.list_blobs(prefix=f"{self.jobs_folder}/{job_id}/"))
        self.logger.info(f"Searching for job {job_id} | prefix={self.jobs_folder}/{job_id}/")
        self.logger.info(f"Found {len(job_blobs)} blobs")
        all_requests = [
            json.loads(blob.download_as_string()) for blob in job_blobs if self.request_fname in blob.name
        ]
        authenticated_requests = [
            req for req in all_requests if user_id == req['user_id'] or user_id in SUPERUSER_IDS
        ]
        if len(all_requests) == 0:
            raise openreview.OpenReviewException('Job not found')
        if len(authenticated_requests) == 0:
            raise openreview.OpenReviewException('Forbidden: Insufficient permissions to access job')
        if len(authenticated_requests) > 1:
            raise openreview.OpenReviewException('Internal Error: Multiple requests found for job')

        ret_list = []
        request = authenticated_requests[0]
        group_group_matching = request.get('entityA', {}).get('type', '') == 'Group' and request.get('entityB', {}).get('type', '') == 'Group'
        paper_paper_matching = request.get('entityA', {}).get('type', '') == 'Note' and request.get('entityB', {}).get('type', '') == 'Note'

        return _get_scores_and_metadata(job_blobs, job_id, group_scoring=group_group_matching, paper_scoring=paper_paper_matching)

        