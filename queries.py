"""
Queries to The Graph db
"""

import json
import requests
import logging
import itertools

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s]: %(message)s')

THE_GRAPH_URL="https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"

# make some standard queries
USERS_QUERY="""
{{
  users(first: {first}, skip: {skip}) {{
    id
  }}
}}
"""


SAMPLE_QUERY="""
{
  uniswapFactories(first: 5) {
    id
    pairCount
    totalVolumeUSD
  }
  tokens(first: 5) {
    id
    symbol
    name
    decimals
  }
}
"""

def get_query_result(query, url=None):
    if url is None:
        url = THE_GRAPH_URL

    headers = {'Content-type' : 'application/json'}
    response = requests.post(
        url,
        data=json.dumps({'query': query}),
        headers=headers)
    return response.text


def get_wallets(batch_size=100):
    """Iterator returning wallets from the users table"""
    skip = 0
    while True:
        query = USERS_QUERY.format(
            first=batch_size,
            skip=skip)
        result = json.loads(get_query_result(query))
        if 'errors' in result:
            raise ValueError("Error executing query")
        else:
            ids = result['data']['users']
            for id_ in ids:
                yield id_['id']


def main():
    # print( get_query_result(SAMPLE_QUERY) )
    for _, id_ in zip(range(10), get_wallets(20)):
        print( id_ )


if __name__ == '__main__':
    main()
