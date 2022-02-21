import argparse
import logging
import os
import time

from jax import random
import jax.numpy as jnp

import numpy as np
import pandas as pd

import pickle

import numpyro
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist
from numpyro.handlers import mask
from numpyro.infer import HMC, MCMC, NUTS

# try working or not with jasa dataset
import awswrangler as wr

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def model_1(sequences, lengths, args, include_prior=True):
    num_sequences, max_length, data_dim = sequences.shape
    with mask(mask=include_prior):
        probs_x = numpyro.sample(
            "probs_x", dist.Dirichlet(0.9 * jnp.eye(args.hidden_dim) + 0.1).to_event(1)
        )
        probs_y_alpha = numpyro.sample(
            "probs_y_alpha", dist.Exponential(0.1).expand([args.hidden_dim, data_dim]).to_event(2)
        )
        probs_y_beta = numpyro.sample(
            "probs_y_beta", dist.Exponential(0.9).expand([args.hidden_dim, data_dim]).to_event(2)
        )
        

    def transition_fn(carry, y):
        x_prev, t = carry
        with numpyro.plate("sequences", num_sequences, dim=-2):
            with mask(mask=(t < lengths)[..., None]):
                x = numpyro.sample(
                    "x",
                    dist.Categorical(probs_x[x_prev]),
                    infer={"enumerate": "parallel"},
                )
                with numpyro.plate("tones", data_dim, dim=-1):
                    numpyro.sample("y", dist.Beta(probs_y_alpha[x.squeeze(-1)],probs_y_beta[x.squeeze(-1)]), obs=y)
        return (x, t + 1), None

    x_init = jnp.zeros((num_sequences, 1), dtype=jnp.int32)
    # NB swapaxes: we move time dimension of `sequences` to the front to scan over it
    scan(transition_fn, (x_init, 0), jnp.swapaxes(sequences, 0, 1))
    
models = {
    name[len("model_") :]: model
    for name, model in globals().items()
    if name.startswith("model_")
}


def main(args):

    model = models[args.model]

    os.environ['AWS_DEFAULT_REGION'] = 'ap-northeast-2'
    contents = wr.athena.read_sql_query('SELECT * FROM kakaowebtoon_kor_content_v2021_11_22', database="ds")
    contents[contents['title']=='나 혼자만 레벨업']

    content_id = 2379

    query = "SELECT DISTINCT user, episode FROM kakaowebtoon_kor_episode_read_v2021_11_22 WHERE content = %s" %str(content_id)
    df = wr.athena.read_sql_query(query, database="ds")

    episodes = np.sort(df.episode.unique())

    from tqdm import tqdm
    rates = []
    for i in tqdm(range(len(episodes)-1)):
        a = df.loc[df.episode==episodes[i]].user.isin(df.loc[df.episode==episodes[i+1]].user)
        rates.append(a.sum()/len(a))
    churnrate = 1 - np.array(rates)
        
    sequences = jnp.asarray(churnrate.reshape(1,-1,1))
    lengths = jnp.array(sequences.shape[1])

    logger.info("-" * 40)
    logger.info("Training {} on {} sequences".format(model.__name__, len(sequences)))

    logger.info("Each sequence has shape {}".format(sequences[0].shape))
    logger.info("Starting inference...")
    rng_key = random.PRNGKey(2)
    start = time.time()
    kernel = {"nuts": NUTS, "hmc": HMC}[args.kernel](model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, sequences, lengths, args=args)
    mcmc.print_summary()
    logger.info("\nMCMC elapsed time: {}".format(time.time() - start))
    
    with open("/home/ec2-user/quality_score/hmm2_ex3_mcmc.pickle", "wb") as f:
        pickle.dump(mcmc, f)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="HMC for HMMs")
    parser.add_argument(
        "-m",
        "--model",
        default="1",
        type=str,
        help="one of: {}".format(", ".join(sorted(models.keys()))),
    )
    parser.add_argument("-n", "--num-samples", nargs="?", default=10, type=int)
    parser.add_argument("-d", "--hidden-dim", default=8, type=int)
    parser.add_argument("--kernel", default="nuts", type=str)
    parser.add_argument("--num-warmup", nargs="?", default=10, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--device", default="gpu", type=str, help='use "cpu" or "gpu".')

    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)