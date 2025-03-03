# API Endpoints
## `POST /expertise`
A valid request body has the following format:
```
{
  "name": string
  "entityA": {
  	"type": string,
	"memberOf": string,
	"id": string,
	"invitation": string,
	"expertise": {
		"exclusion": { "invitation": string }
	}
  },
  "entityB": {
  	"type": string,
	"memberOf": string,
	"id": string,
	"invitation": string,
	"expertise": {
		"exclusion": { "invitation": string }
	}
  },
  "model": {
  	"name": string,
	"sparseValue": int,
	"useTitle": boolean,
	"useAbstract": boolean,
	"scoreComputation": "avg" or "max",
	"skipSpecter": boolean
  }
}
```

Returns code `200` on successful job submission with: `{"job_id": "string"}`\
Returns code `400` if there was an error in the submitted config

## `GET /expertise/status`
This endpoint gets the status of a single job with the given `job_id`. A valid request body has the following format:
```
{"job_id": string}
```

Returns the status, if any, that were submitted by the user and has the job id:
```
{
	"job_id": string,
	"name": string,
	"status": string,
	"description": string,
	"config": {...}
}
```

## `GET /expertise/status/all`
This endpoint gets the status of all jobs submitted by the user. A valid request body has the following format:
```
{}
```

Returns a list of jobs, if any, that were submitted by the user and has the job id if provided:
```
{
  "results": [
      {
        "job_id": string,
        "name": string,
        "status": string,
        "description": string,
        "config": {...}
      }
  ]
}
```

## `GET /expertise/results`
This endpoint retrieves the results of a job with the matching job ID and optionally removes the scores from the server after retrieval. A valid request body has the following format:
```
{
  "job_id": string,
  "deleteOnGet": false [Optional],
}
```

Returns code `200` if the scores were successfully retrieved\
Returns code `404` if there were no scores found for the current job \
Returns code `403` if attempting to access a job that the user does not have access to

The scores are returned in the body of the response with the following format
```
{
  "results": {
      "submission": string,
      "user": string,
      "score": float
  }
}
```
# Interpreting Status and Descriptions
The API gives the user the ability to query the status of their job at several stages in processing. The following are the `(status, description)` pairs supported by the API:
1. **Initialized**: Server received config and allocated space
2. **Queued**: Job is waiting to start fetching OpenReview data
3. **Fetching Data**: Job is currently fetching data from OpenReview
4. **Queued for Expertise**: Job has assembled the data and is waiting in queue for the expertise model
5. **Running Expertise**: Job is running the selected expertise model to compute scores
6. **Completed**: Job is complete and the computed scores are ready

# Redis Search Optimization

The service now includes optimized Redis search capabilities to improve job lookup efficiency.

## Redis JSON Migration Guide

The system now supports Redis JSON for more efficient storage and retrieval of job data. To use this feature:

1. Install Redis JSON module:
   ```
   # For Docker/Redis Stack
   docker run -p 6379:6379 redis/redis-stack
   
   # For standalone Redis
   # See https://redis.io/docs/stack/json/ for installation instructions
   ```

2. Migrate existing data:
   ```
   python -m expertise.service migrate --config=/path/to/config.cfg
   ```

3. Verify the migration:
   ```
   python -m expertise.service migrate --config=/path/to/config.cfg
   ```
   This will show statistics after the migration.

## Benefits of Redis JSON

- **Faster job retrieval**: Jobs are now indexed by user and creation date
- **More efficient searching**: No need to deserialize all jobs when searching
- **Better sorted results**: Results are pre-sorted by creation date
- **Reduced memory usage**: No need to keep all job objects in memory

## Technical Implementation

The Redis database now uses:

1. **JSON storage**: Jobs are stored as JSON strings instead of pickled objects
2. **Sorted Sets**: User-to-job mappings are stored in sorted sets with creation date as score
3. **Automatic sorting**: Results are automatically sorted by creation date

This implementation significantly improves performance when:
- Listing all jobs for a user
- Searching for specific jobs
- Retrieving job details

## Compatibility

The system maintains backward compatibility with systems that don't have Redis JSON installed. In those cases, it will:
- Fall back to pickle serialization
- Perform linear scans for job retrieval
- Sort results manually after retrieval

## Monitoring

A new stats function has been added for monitoring Redis usage:
```python
from expertise.service.utils import RedisDatabase

db = RedisDatabase(...)
stats = db.get_stats()
print(stats)
```

This will show:
- Total job count
- Users count
- Jobs per user
- Redis memory usage
- Redis JSON module availability

