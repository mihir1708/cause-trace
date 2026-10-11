"""Load test for POST /v1/search (run headless by `make loadtest`, see loadtests/run.py).

Each simulated user sends incident-style queries with a short think time. LOAD_MODE picks
the search mode (default rrf_rerank, what the API defaults to).
"""

import os
import random

from locust import HttpUser, between, task

MODE = os.environ.get("LOAD_MODE", "rrf_rerank")
QUERIES = (
    "engine lost power shortly after takeoff and the pilot made a forced landing in a field",
    "the airplane encountered a gusting crosswind during landing and departed the runway",
    "fuel exhaustion during cruise flight resulted in a total loss of engine power",
    "the student pilot bounced the landing and the nose landing gear collapsed",
    "carburetor icing conditions and a partial loss of engine power",
    "the helicopter experienced a loss of tail rotor effectiveness while hovering",
    "the pilot continued visual flight into instrument meteorological conditions",
    "a main landing gear brake failure caused a loss of directional control on landing",
    "the glider struck trees during an off-airport landing after losing lift",
    "inadequate preflight inspection did not detect water contamination in the fuel",
)


class SearchUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def search(self) -> None:
        self.client.post(
            "/v1/search",
            json={
                "query": random.choice(QUERIES),
                "before_date": "2024-01-01",
                "top_k": 10,
                "mode": MODE,
            },
            name="/v1/search",
        )
