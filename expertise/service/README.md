# API Endpoints
## `POST /expertise`
This endpoint accepts a subset of fields that would be expected in a `config.json` file as described in the `README.md` of the root directory. A valid request body has the following format:
```
{
  "name": "string",
  "match_group": "string",
  "paper_invitation": "string",
  "exclusion_inv": "string" [Optional],
  "model": "specter+mfr" [Optional],
  "model_params" [Optional]: {
      "use_title": true,
      "use_abstract": true,
      "average_score": false,
      "max_score": true,
      "skip_specter": false
  }
}
```

Returns code `200` on successful job submission with: `{"job_id": "string"}`\
Returns code `400` if there was an error in the submitted config

## `GET /expertise/status`
This endpoint gets the status of all jobs submitted by the user, or a single job with the given `job_id`. A valid request body has the following format:
```
{"id": "string" [Optional]}
```

Returns a list of jobs, if any, that were submitted by the user and has the job id if provided:
```
{
  "results": [
      {
        "job_id": "string",
        "name": "string",
        "status": "string",
        "description": "string"
      }
  ]
}
```

## `GET /expertise/results`
This endpoint retrieves the results of a job with the matching job ID and optionally removes the scores from the server after retrieval. A valid request body has the following format:
```
{
  "id": "string",
  "delete_on_get": false [Optional],
}
```

Returns code `200` if the scores were successfully retrieved\
Returns code `404` if there were no scores found for the current job \
Returns code `403` if attempting to access a job that the user does not have access to

The scores are returned in the body of the response with the following format
```
{
  "results": {
      "submission": "string",
      "user": "string",
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

