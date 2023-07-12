# Credit card dataset testing on Vertex AI
## Steps to replicate issue:
1.  Run to package the source distribution file for training job: <br />```python setup.py sdist --formats=gztar``` 
2.  Change directory to dist folder and run command to copy dist file to GCS bucket: <br />```gsutil cp trainer-0.1.tar.gz gs://preprocessed_credit_card_data```
3.  Run all cells in trainerJob.ipynb to start training job, to which error would then be replicated (link to logs explorer will be at the bottom of the cell when cell finishes running or when error is displayed in notebook)

## Error:
```
Traceback (most recent call last):
  File "/root/.local/lib/python3.7/site-packages/google/api_core/grpc_helpers.py", line 72, in error_remapped_callable
    return callable_(*args, **kwargs)
  File "/root/.local/lib/python3.7/site-packages/grpc/_channel.py", line 1030, in __call__
    return _end_unary_response_blocking(state, call, False, None)
  File "/root/.local/lib/python3.7/site-packages/grpc/_channel.py", line 910, in _end_unary_response_blocking
    raise _InactiveRpcError(state)  # pytype: disable=not-instantiable
grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with:
	status = StatusCode.PERMISSION_DENIED
	details = "request failed: the user does not have 'bigquery.readsessions.create' permission for 'projects/203287538440'"
	debug_error_string = "UNKNOWN:Error received from peer ipv4:74.125.69.95:443 {grpc_message:"request failed: the user does not have \'bigquery.readsessions.create\' permission for \'projects/203287538440\'", grpc_status:7, created_time:"2023-07-12T02:38:33.411291269+00:00"}"
```
