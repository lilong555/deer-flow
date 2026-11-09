# Thread-operation release recovery

The probe exercises `RunManager` and `RunRepository` against PostgreSQL 18.1. Independent connections hold real row locks during release. Cancellation waits until `pg_blocking_pids` confirms the blocked SQL request.

| Source | Passed | Failed |
| --- | ---: | ---: |
| Baseline `4e09e63a68f55bf5203838598b399fd03b2e80bd` | 7 | 20 |
| Candidate `a3f6d8d84e67585bfd3435fe03f7723be053b94a` | 27 | 0 |

Cases cover all five internal operation kinds, reject/interrupt/rollback, continuing lock contention, original body exceptions, cancellation, captured users, peer reservations, new idempotency keys and evicted completed-key replay. Each case creates and removes its own UUID schema.

`baseline.json` and `candidate.json` contain individual outcomes, source identities and the SHA-256 of the identical probe used for both revisions.

To reproduce, keep this directory outside the source checkout, check out either recorded revision, and install its backend dependencies with `uv sync --frozen --extra postgres --python 3.12`. Start an isolated loopback database matching `postgres-fixture.json`, for example:

```bash
docker run --rm -d --name deerflow-reservation-pg \
  -e POSTGRES_USER=deerflow_test -e POSTGRES_HOST_AUTH_METHOD=trust \
  -p 127.0.0.1:55441:5432 postgres:18.1
```

From the source checkout's `backend/` directory, run:

```bash
.venv/bin/python /path/to/thread-operation-recovery/postgres-recovery.py \
  --output /tmp/postgres-release-results.json
```

The probe asserts that the imported manager belongs to the current checkout. A nonzero exit on the recorded baseline represents the reproduced failures. The candidate exits successfully. Remove the test container with `docker rm -f deerflow-reservation-pg` when finished.
