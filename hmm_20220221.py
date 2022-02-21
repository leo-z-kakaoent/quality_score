import os
import awswrangler as wr
import numpy as np
import pandas as pd

import jax
import numpy as np

from jax import lax, random
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.distributions import constraints

from tqdm import tqdm

from random import shuffle


os.environ['AWS_DEFAULT_REGION'] = 'ap-northeast-2'

contents = wr.athena.read_sql_query('SELECT * FROM kakaowebtoon_kor_content_v2021_11_22', database="ds")

contents[contents['title']=='나 혼자만 레벨업']

content_id = 2320

query = "SELECT user, episode FROM kakaowebtoon_kor_episode_read_v2021_11_22 WHERE content = %s" %str(content_id)

df = wr.athena.read_sql_query(query, database="ds")

all_unique_users = len(df.drop_duplicates('user'))

def aggregate_fn(x):
    return len(np.unique(x))/all_unique_users

episode_userrate = df.groupby('episode').aggregate(lambda x: aggregate_fn(x))

def forward_one_step(prev_log_prob, curr_y, transition_log_prob, emission_fn):
    log_prob_tmp = jnp.expand_dims(prev_log_prob, axis=1) + transition_log_prob
    log_prob = log_prob_tmp + emission_fn.log_prob(curr_y)
    return logsumexp(log_prob, axis=0)


def forward_log_prob(init_log_prob, ys, transition_log_prob, emission_fn):
    def scan_fn(log_prob, y):
        return (
            forward_one_step(log_prob, y, transition_log_prob, emission_fn),
            None,  # we don't need to collect during scan
        )
    log_prob, _ = lax.scan(scan_fn, init_log_prob, ys)
    return log_prob

def hmm(ys, hidden_dim):
    xt_prob = numpyro.sample("xt_prob", dist.Dirichlet(jnp.eye(hidden_dim)+0.5))
    temp_params = jnp.asarray(np.random.random([hidden_dim,2]))
    emission_prob = numpyro.sample('emission_prob', dist.Beta(temp_params[:,0], temp_params[:,1]))

    transition_log_prob = jnp.log(xt_prob)
    init_log_prob = jnp.zeros((hidden_dim), dtype=jnp.float32)
    log_prob = forward_log_prob(
        init_log_prob,
        ys,
        transition_log_prob,
        emission_prob,
    )
    log_prob = logsumexp(log_prob, axis=0, keepdims=True)
    numpyro.factor("forward_log_prob", log_prob)