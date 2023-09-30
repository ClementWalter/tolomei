# %% Imports
import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from starknet_py.contract import Call, Contract
from starknet_py.hash.selector import get_selector_from_name

from src.utils.constants import RPC_CLIENT
from src.utils.starknet import get_starknet_account

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# %% Ekubo prices
TOKENS = {
    "0xda114221cb83fa859dbdb4c44beeaa0bb37c7537ad5ae66fe5e0efd20e6eb3": "DAI",
    "0x124aeb495b947201f5fac96fd1138e326ad86195b98df6dec9009158a533b49": "LORDS",
    "0x319111a5037cbec2b3e638cc34a3474e2d2608299f3e62866e9cc683208c610": "rETH",
    "0x3fe2b97c1fd336e750087d68b9b867997fd64a2661ff3ca5a7c771641e8e7ac": "WBTC",
    "0x42b8f0484674ca266ac5d08e4ac6a3fe65bd3129795def2dca5c34ecc5f96d2": "wstETH",
    "0x53c91253bc9682c04929ca02ed00b3e423f6710d2ee7e0d5ebb06f3ecf368a8": "USDC",
    "0x68f5c6a61780768455de69077e07e89787839bf8166decfbf92b645209c0fb8": "USDT",
    "0x49d36570d4e46f48e99674bd3fcc84644ddd6b96f7c741b1562b82f9e004dc7": "ETH",
}
TOKEN_NAME_TO_ADDRESS = {value: key for key, value in TOKENS.items()}

TICK_SPACING = {
    5982: "0.3% / 0.6%",
    200: "0.01% / 0.02%",
    1000: "0.05% / 0.1%",
}
STABLE = ["USDC", "DAI", "USDT"]
EKUBO_CORE_ADDRESS = (
    "0x00000005dd3d2f4429af886cd1a3b08289dbcea99a294197e9eb43b0e0325b4b"
)

prices_df = (
    pd.concat(
        [
            pd.DataFrame(
                requests.get(f"https://mainnet-api.ekubo.org/price/{address}").json()[
                    "prices"
                ]
            ).assign(base=name)
            for address, name in TOKENS.items()
        ]
    )
    .replace({"token": TOKENS})
    .reset_index(drop=True)
    .pivot(index="token", columns=["base"], values="price")
    .astype(float)
    .fillna(0)
)
account = await get_starknet_account()


# %% Search algorithm
def find_arbitrage_for_token(origin, prices):
    swap_cost = 0.05 / 100
    steps = set(list(range(len(prices)))) - {origin}

    route = list(steps)
    best_route = route
    full_route = [origin, *route, origin]
    best_prices = [prices.values[o, d] for o, d in zip(full_route[:-1], full_route[1:])]
    best_profit = np.prod(best_prices) * (1 - swap_cost) ** len(best_prices)

    for _ in range(2000):
        i = np.random.randint(len(route))
        proba = np.random.rand()
        # with p = 1/2 we remove a node, or we swap
        if proba > 0.66 and len(route) > 1:
            route = route[0:i] + route[i + 1 :]
        elif proba > 0.33 and len(route) < len(steps):
            new_index = np.random.choice(list(steps - set(route)))
            route = route + [new_index]
        else:
            if i == len(route) - 1:
                i -= 1
            route[i], route[i + 1] = route[i + 1], route[i]

        full_route = [origin, *route, origin]
        prices_tmp = [
            prices.values[o, d] for o, d in zip(full_route[:-1], full_route[1:])
        ]
        profit = np.prod(prices_tmp) * (1 - swap_cost) ** len(prices_tmp)

        if profit > best_profit:
            best_route = [*route]
            best_profit = profit
            best_prices = prices_tmp

    return (
        (
            [prices.columns[origin]]
            + [prices.columns[i] for i in best_route]
            + [prices.columns[origin]]
        ),
        best_profit,
        best_prices,
    )


# %% Run on given prices matrix
arbitrages = pd.DataFrame(
    [find_arbitrage_for_token(i, prices_df) for i in range(len(prices_df))],
    columns=["route", "profit", "prices"],
).sort_values("profit", ascending=False)
logger.info(f"Arbitrages:\n{arbitrages}")


# %% Fetch pools data
async def get_pool_price(pool):
    logger.info(
        f"Fetching pool price for {TOKENS[pool['token_from']]}/{TOKENS[pool['token_to']]}"
    )
    token_0, token_1 = (
        (int(pool["token_from"], 16), int(pool["token_to"], 16))
        if int(pool["token_to"], 16) > int(pool["token_from"], 16)
        else (int(pool["token_to"], 16), int(pool["token_from"], 16))
    )
    (
        sqrt_ratio_low,
        sqrt_ratio_high,
        tick_mag,
        tick_sign,
        *call_points,
    ) = await RPC_CLIENT.call_contract(
        Call(
            to_addr=int(EKUBO_CORE_ADDRESS, 16),
            selector=get_selector_from_name("get_pool_price"),
            calldata=[
                token_0,
                token_1,
                int(pool["fee"]),
                int(pool["tick_spacing"]),
                int(pool["extension"]),
            ],
        )
    )
    sqrt_ratio = sqrt_ratio_high + sqrt_ratio_low / 2**128
    price = sqrt_ratio * sqrt_ratio
    return price if token_0 == int(pool["token_from"], 16) else 1 / price


def get_pool(token_from, token_to):
    logger.info(f"Fetching pools for pair {TOKENS[token_from]}/{TOKENS[token_to]}")
    response = requests.get(
        f"https://mainnet-api.ekubo.org/pair/{token_from}/{token_to}"
    )
    return pd.DataFrame(response.json()["topPools"]).assign(
        token_from=token_from, token_to=token_to
    )


arbitrage = arbitrages.iloc[0]
pools = (
    pd.concat(
        [
            get_pool(TOKEN_NAME_TO_ADDRESS[token_0], TOKEN_NAME_TO_ADDRESS[token_1])
            for token_0, token_1 in zip(arbitrage.route[:-1], arbitrage.route[1:])
        ]
    )
    .reset_index(drop=True)
    .loc[lambda df: df.tick_spacing.astype(int).isin(TICK_SPACING.keys())]
)


selected_pools = (
    pools.assign(
        price=[await get_pool_price(pool) for pool in pools.to_dict("records")]
    )
    .groupby(by=["token_from", "token_to"])
    .apply(lambda group: group.loc[lambda df: df.volume0_24h.idxmax()])
    .reset_index(drop=True)
    .set_index("token_from")
    .loc[[TOKEN_NAME_TO_ADDRESS[token] for token in arbitrage.route[:-1]]]
    .reset_index()
    .assign(required_liquidity=lambda df: df.price.cumprod())
    .assign(
        token_from=lambda df: df.token_from.map(lambda address: int(address, 16)),
        token_to=lambda df: df.token_to.map(lambda address: int(address, 16)),
        token_0=lambda df: np.minimum(df.token_from, df.token_to),
        token_1=lambda df: np.maximum(df.token_from, df.token_to),
    )
)
(
    selected_pools.filter(
        items=[
            "token_from",
            "token_to",
            "fee",
            "price",
        ]
    )
    .assign(
        token_from=lambda df: df.token_from.map(hex),
        token_to=lambda df: df.token_to.map(hex),
    )
    .replace({"token_from": TOKENS, "token_to": TOKENS})
)

if selected_pools.price.prod() < 1:
    logger.error("Final route is losing money")
else:
    logger.info(f"Actual profit: {selected_pools.price.prod()}")

# %% Send tx
swap_params = (
    selected_pools.reindex(
        [
            "token_0",
            "token_1",
            "fee",
            "tick_spacing",
            "extension",
            "token_from",
            "token_to",
        ],
        axis=1,
    )
    .assign(
        token_0=lambda df: df.token_0.map(int),
        token_1=lambda df: df.token_1.map(int),
        fee=lambda df: df.fee.map(int),
        tick_spacing=lambda df: df.tick_spacing.map(int),
        extension=lambda df: df.extension.map(int),
        token_from=lambda df: df.token_from.map(int),
        token_to=lambda df: df.token_to.map(int),
    )
    .rename(columns={"token_0": "token0", "token_1": "token1"})
    .assign(
        pool_key=lambda df: df[
            [
                "token0",
                "token1",
                "fee",
                "extension",
                "tick_spacing",
            ]
        ].to_dict("records")
    )
    .assign(
        route=lambda df: df[["token_from", "token_to", "pool_key"]].to_dict("records")
    )
    .route.to_list()
)


# %% Send tx
balance_low, balance_high = await RPC_CLIENT.call_contract(
    Call(
        to_addr=TOKEN_NAME_TO_ADDRESS["ETH"],
        selector=get_selector_from_name("balanceOf"),
        calldata=[account.address],
    )
)
balance = balance_low + balance_high * 2**128
base_price_in_usd = prices_df.loc[arbitrage.route[0]]["USDC"]
amount_from_in_usd = 1

flashswap = await Contract.from_address(
    0x03E5538F146CCC90EAB5B60B374123EB54D97621879A3392BAA1BD12CE0BF3FF, account
)
await flashswap.functions["flashloan_swap"].invoke(
    flashswap_params={
        "amount_from": int(amount_from_in_usd / base_price_in_usd),
        "routes": swap_params,
    },
    max_fee=balance,
)

# %%
