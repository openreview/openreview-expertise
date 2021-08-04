# Running the Server
The server is implemented in Flask and can be started from the command line:
```
python -m expertise.service --host localhost --port 5000
```

By default, the app will run on `http://localhost:5000`. The endpoint `/expertise/test` should show a simple page indicating that Flask is running.

## Accessing the Endpoints
Access to all endpoints requires authentication in the header of the request.

**`POST /expertise`**

**Request Body:**

```
{
  config: JSON object containing options for both create_dataset and run_expertise
}
```

**Returns:**

`{ user_id: str, job_id: int }`

---

**`GET /jobs`**

**Request Body:**

`{}`

**Returns:**

```
{
  'dataset': [
    {
      job_name: str,
      job_id: int,
      status: str,
    }, ...
  ],
  'expertise': [
    {
      job_name: str,
      job_id: int,
      status: str,
    }, ...
  ]
}
```

---

**`GET /results`**

**Request Body:**

`{ job_id: int, delete_on_get: boolean }`

**Returns:**

```
{
  results: [
    {
      submission: str,
      user: str,
      status: float,
    }, ...
  ]
}
```

# Basic Queue Structures
## JobQueue

JobData is a dataclass that's used to store metadata about each jobs and it is defined with the following fields (from `queue.py`):
```
    id: str = field(
        metadata={"help": "The profile id at the time of submission"},
    )
    job_name: str = field(
        metadata={"help": "The name of the job specified in the submitted config file"},
    )
    config: dict = field(
        metadata={"help": "The submitted configuration file as a dictionary"},
    )
    timeout: int = field(
        default=0,
        metadata={"help": "The maximum amount of time to run this job"},
    )
```

JobQueue is an abstract wrapper for the built-in Python `queue.Queue()` class. The built-in queue offers a standard FIFO queue that allows both asynchronous and blocking calls for `put()` and `get()`. JobQueue works under the assumption that each job accepts a configuration file and makes modifications to the filesystem to store intermediate and final results. JobQueue offers a set of services on top the built-in queue including:
1.  Keeping track of a history of submitted jobs
2.  Pre-emptive canceling of jobs that are still in queue
3.  Timing out of possibly long running jobs
4.  Retrieving the current status of an arbitrary job
5.  Multi-threaded multi-process handling of submitted jobs
6.  Allocating a directory on a per-job basis to enforce job isolation

JobQueue defines several status codes that are free to asynchronously query during a job's lifetime:
- `queued` -- The job is currently awaiting processing by a worker
- `processing` -- The job is currently being worked on by a worker
- `completed` -- The job has finished and the results are currently stored on the server
- `stale` -- The job has been canceled before it arrived at processing
- `timeout` -- The job has exceeded the specified timeout (there is no timeout by default)
- `error` -- While processing, the job process has encountered an error

In order to use the JobQueue class, create a subclass of JobQueue that implements the following functions
1.  `get_result(self, user_id: str, delete_on_get: bool = True, job_id: str = '', job_name: str = '') -> List[dict]`:

This function must read the server's filesystem using the configuation dictionary and return the results in a JSON object. Optional: Clean up the directory depending on the `delete_on_get` flag. Generally, these functions must 1) fetch the job data, 2) collect results from reading created files, and 3) handle cleaning up the directory.

2.  `def run_job(self, config: dict) -> None`:

This function performs the actual work given a configuration dictionary. Once the corresponding job begins processing, this function gets called in a separate process.

3. [Optional] `def _prepare_job(self, job_info: JobData) -> None`:

This function serves as a pre-processing step for all job requests sent to the queue. By default, this function performs no modifications to the job info. However, this function may need to be overridden in order to adjust some configuration parameters to fit the server's filesystem.

## TwoStepQueue

TwoStepQueue is a subclass of JobQueue that implements `get_result`. This type of queue serves the same kind of function as JobQueue, in that it maintains a queue of jobs and executes jobs - but additionally contains an `inner_queue` instance variable. Once a job finishes processing in the TwoStepQueue, it immediately enqueues the same JobData object to the inner queue. The use case for this is that two distinct tasks can share a causal relationship where the first task is time-sensitive (for example, in the case executing a job before a login token expires).

Any queries for information about the jobs on this type of queue are typically returned in a JSON/dictionary object that contains fields for the first and second task. By default, `get_result` of this type of queue assumes that the first task does not store meaningful results and defers to the inner queue for handling result requests.

# OpenReview Queues

First, a subclass of JobData is created that adds additonal optional fields `token` and `baseurl` to support querying of an API server for OpenReview data.

## ExpertiseQueue

This queue is a basic JobQueue that does no additional pre-processing on submitted job requests, computes affinity scores given a retrieved dataset, and attempts to read the score CSV stored in the `scores_path` parameter of the config and creates JSON objects with fields:
1.  `submission`
2.  `user`
3.  `score`

## UserPaperQueue

This queue is a TwoStepQueue that performs the following tasks: dataset creation and affinity scoring between submissions and authors. The actual work performed by the first task queue is assembling the dataset using credentials provided by the job request. This queue also implements `_prepare_job` in order to override directory fields in the config file to be isolated from other jobs.
