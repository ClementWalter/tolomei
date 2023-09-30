import functools
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Union, cast

import requests
from starknet_py.contract import Contract
from starknet_py.hash.selector import get_selector_from_name
from starknet_py.net.account.account import Account
from starknet_py.net.client_models import Call, TransactionStatus
from starknet_py.net.signer.stark_curve_signer import KeyPair

from src.utils.constants import (
    BUILD_DIR,
    BUILD_DIR_FIXTURES,
    CONTRACTS,
    CONTRACTS_FIXTURES,
    DEPLOYMENTS_DIR,
    ETH_TOKEN_ADDRESS,
    GATEWAY_CLIENT,
    NETWORK,
    RPC_CLIENT,
    SOURCE_DIR,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Due to some fee estimation issues, we skip it in all the calls and set instead
# this hardcoded value. This has no impact apart from enforcing the signing wallet
# to have at least 0.1 ETH
_max_fee = int(5e15)


def int_to_uint256(value):
    value = int(value)
    low = value & ((1 << 128) - 1)
    high = value >> 128
    return {"low": low, "high": high}


async def get_starknet_account(
    address=None,
    private_key=None,
) -> Account:
    address = address or NETWORK["account_address"]
    if address is None:
        raise ValueError(
            "address was not given in arg nor in env variable, see README.md#Deploy"
        )
    address = int(address, 16)
    private_key = private_key or NETWORK["private_key"]
    if private_key is None:
        raise ValueError(
            "private_key was not given in arg nor in env variable, see README.md#Deploy"
        )
    key_pair = KeyPair.from_private_key(int(private_key, 16))

    public_key = None
    for selector in ["get_public_key", "getPublicKey", "getSigner", "get_owner"]:
        try:
            call = Call(
                to_addr=address,
                selector=get_selector_from_name(selector),
                calldata=[],
            )
            public_key = (
                await RPC_CLIENT.call_contract(call=call, block_hash="latest")
            )[0]
            break
        except Exception as err:
            if (
                err.message == "Client failed with code 40: Contract error."
                or err.message
                == "Client failed with code 21: Invalid message selector."
                or "StarknetErrorCode.ENTRY_POINT_NOT_FOUND_IN_CONTRACT" in err.message
                or err.message.find("Client failed with code -32603") != -1
            ):
                continue
            else:
                logger.error(f"Raising for account at address {hex(address)}")
                raise err

    if public_key is not None:
        if key_pair.public_key != public_key:
            raise ValueError(
                f"Public key of account 0x{address:064x} is not consistent with provided private key"
            )
    else:
        logger.warning(
            f"⚠️  Unable to verify public key for account at address 0x{address:x}"
        )

    return Account(
        address=address,
        client=RPC_CLIENT,
        chain=NETWORK["chain_id"],
        key_pair=key_pair,
    )


async def get_eth_contract() -> Contract:
    # TODO: use .from_address when katana implements getClass
    return Contract(
        ETH_TOKEN_ADDRESS,
        json.loads((Path("scripts") / "utils" / "erc20.json").read_text())["abi"],
        await get_starknet_account(),
    )


async def get_contract(contract_name) -> Contract:
    # TODO: use .from_address when katana implements getClass
    return Contract(
        get_deployments()[contract_name]["address"],
        json.loads(get_artifact(contract_name).read_text())["abi"],
        await get_starknet_account(),
    )


async def fund_address(address: Union[int, str], amount: float):
    """
    Fund a given starknet address with {amount} ETH.
    """
    address = int(address, 16) if isinstance(address, str) else address
    amount = amount * 1e18
    if NETWORK["name"] == "starknet-devnet":
        response = requests.post(
            "http://127.0.0.1:5050/mint",
            json={"address": hex(address), "amount": amount},
        )
        if response.status_code != 200:
            logger.error(f"Cannot mint token to {address}: {response.text}")
        logger.info(f"{amount / 1e18} ETH minted to {hex(address)}")
    else:
        account = await get_starknet_account()
        eth_contract = await get_eth_contract()
        balance = (await eth_contract.functions["balanceOf"].call(account.address)).balance  # type: ignore
        if balance < amount:
            raise ValueError(
                f"Cannot send {amount / 1e18} ETH from default account with current balance {balance / 1e18} ETH"
            )
        prepared = eth_contract.functions["transfer"].prepare(
            address, int_to_uint256(amount)
        )
        tx = await prepared.invoke(max_fee=_max_fee)

        status = await wait_for_transaction(tx.hash)
        status = "✅" if status == TransactionStatus.ACCEPTED_ON_L2 else "❌"
        logger.info(
            f"{status} {amount / 1e18} ETH sent from {hex(account.address)} to {hex(address)}"
        )
        balance = (await eth_contract.functions["balanceOf"].call(address)).balance  # type: ignore
        logger.info(f"💰 Balance of {hex(address)}: {balance / 1e18}")


def dump_declarations(declarations):
    json.dump(
        {name: hex(class_hash) for name, class_hash in declarations.items()},
        open(DEPLOYMENTS_DIR / "declarations.json", "w"),
        indent=2,
    )


def get_declarations():
    return {
        name: int(class_hash, 16)
        for name, class_hash in json.load(
            open(DEPLOYMENTS_DIR / "declarations.json")
        ).items()
    }


def dump_deployments(deployments):
    json.dump(
        {
            name: {
                **deployment,
                "address": hex(deployment["address"]),
                "tx": hex(deployment["tx"]),
                "artifact": str(deployment["artifact"]),
            }
            for name, deployment in deployments.items()
        },
        open(DEPLOYMENTS_DIR / "deployments.json", "w"),
        indent=2,
    )


def get_deployments():
    try:
        return json.load(open(DEPLOYMENTS_DIR / "deployments.json", "r"))
    except FileNotFoundError:
        return {}


def get_artifact(contract_name):
    is_fixture = is_fixture_contract(contract_name)
    return (
        BUILD_DIR / f"{contract_name}.json"
        if not is_fixture
        else BUILD_DIR_FIXTURES / f"{contract_name}.json"
    )


def get_tx_url(tx_hash: int) -> str:
    return f"{NETWORK['explorer_url']}/tx/0x{tx_hash:064x}"


def is_fixture_contract(contract_name):
    return CONTRACTS_FIXTURES.get(contract_name) is not None


def compile_contract(contract):
    is_fixture = is_fixture_contract(contract["contract_name"])
    contract_build_path = get_artifact(contract["contract_name"])

    output = subprocess.run(
        [
            "starknet-compile-deprecated",
            CONTRACTS[contract["contract_name"]]
            if not is_fixture
            else CONTRACTS_FIXTURES[contract["contract_name"]],
            "--output",
            contract_build_path,
            "--cairo_path",
            str(SOURCE_DIR),
            *(["--no_debug_info"] if not NETWORK["devnet"] else []),
            *(["--account_contract"] if contract["is_account_contract"] else []),
            *(
                ["--disable_hint_validation"]
                if NETWORK["name"] == "starknet-devnet"
                else []
            ),
        ],
        capture_output=True,
    )
    if output.returncode != 0:
        raise RuntimeError(output.stderr)

    def _convert_offset_to_hex(obj):
        if isinstance(obj, list):
            return [_convert_offset_to_hex(i) for i in obj]
        if isinstance(obj, dict):
            return {key: _convert_offset_to_hex(obj[key]) for key, value in obj.items()}
        if isinstance(obj, int) and obj >= 0:
            return hex(obj)
        return obj

    compiled = json.loads(contract_build_path.read_text())
    compiled = {
        **compiled,
        "entry_points_by_type": _convert_offset_to_hex(
            compiled["entry_points_by_type"]
        ),
    }
    json.dump(
        compiled,
        open(
            contract_build_path,
            "w",
        ),
        indent=2,
    )


async def deploy(contract_name, *args):
    logger.info(f"ℹ️  Deploying {contract_name}")
    abi = json.loads(Path(get_artifact(contract_name)).read_text())["abi"]
    account = await get_starknet_account()

    deploy_result = await Contract.deploy_contract(
        account=account,
        class_hash=get_declarations()[contract_name],
        abi=abi,
        constructor_args=list(args),
        max_fee=_max_fee,
    )
    status = await wait_for_transaction(deploy_result.hash)
    status = "✅" if status == TransactionStatus.ACCEPTED_ON_L2 else "❌"
    logger.info(
        f"{status} {contract_name} deployed at: {hex(deploy_result.deployed_contract.address)}"
    )
    return {
        "address": deploy_result.deployed_contract.address,
        "tx": deploy_result.hash,
        "artifact": get_artifact(contract_name),
    }


async def invoke_address(contract_address, function_name, *calldata, account=None):
    account = account or (await get_starknet_account())
    logger.info(
        f"ℹ️  Invoking {function_name}({json.dumps(calldata) if calldata else ''}) "
        f"at address {hex(contract_address)[:10]}"
    )
    return await account.execute(
        Call(
            to_addr=contract_address,
            selector=get_selector_from_name(function_name),
            calldata=cast(List[int], calldata),
        ),
        max_fee=_max_fee,
    )


async def invoke_contract(
    contract_name, function_name, *inputs, address=None, account=None
):
    account = account or (await get_starknet_account())
    deployments = get_deployments()
    contract = Contract(
        deployments[contract_name]["address"] if address is None else address,
        json.load(open(get_artifact(contract_name)))["abi"],
        account,
    )
    call = contract.functions[function_name].prepare(*inputs, max_fee=_max_fee)
    logger.info(
        f"ℹ️  Invoking {contract_name}.{function_name}({json.dumps(inputs) if inputs else ''})"
    )
    return await account.execute(call, max_fee=_max_fee)


async def invoke(contract, *args, **kwargs):
    response = await (
        invoke_address(contract, *args, **kwargs)
        if isinstance(contract, int)
        else invoke_contract(contract, *args, **kwargs)
    )
    logger.info(f"⏳ Waiting for tx {get_tx_url(response.transaction_hash)}")
    status = await wait_for_transaction(response.transaction_hash)
    status = "✅" if status == TransactionStatus.ACCEPTED_ON_L2 else "❌"
    logger.info(
        f"{status} {contract}.{args[0]} invoked at tx: %s",
        hex(response.transaction_hash),
    )
    return response.transaction_hash


async def call_address(contract_address, function_name, *calldata):
    account = await get_starknet_account()
    return await account.client.call_contract(
        Call(
            to_addr=contract_address,
            selector=get_selector_from_name(function_name),
            calldata=cast(List[int], calldata),
        )
    )


async def call_contract(contract_name, function_name, *inputs, address=None):
    deployments = get_deployments()
    account = await get_starknet_account()
    contract = Contract(
        deployments[contract_name]["address"] if address is None else address,
        json.load(open(get_artifact(contract_name)))["abi"],
        account,
    )
    return await contract.functions[function_name].call(*inputs)


async def call(contract, *args, **kwargs):
    return await (
        call_address(contract, *args, **kwargs)
        if isinstance(contract, int)
        else call_contract(contract, *args, **kwargs)
    )


# TODO: use RPC_CLIENT when RPC wait_for_tx is fixed, see https://github.com/kkrt-labs/kakarot/issues/586
# TODO: Currently, the first ping often throws "transaction not found"
@functools.wraps(RPC_CLIENT.wait_for_tx)
async def wait_for_transaction(*args, **kwargs):
    """
    We need to write this custom hacky wait_for_transaction instead of using the one from starknet-py
    because the RPCs don't know RECEIVED, PENDING and REJECTED states currently.
    """
    if GATEWAY_CLIENT is not None:
        # Gateway case, just use it
        receipt = await GATEWAY_CLIENT.wait_for_tx(*args, **kwargs)
        return receipt.status

    start = datetime.now()
    elapsed = 0
    check_interval = kwargs.get("check_interval", NETWORK.get("check_interval", 15))
    max_wait = kwargs.get("max_wait", NETWORK.get("max_wait", 30))
    transaction_hash = args[0] if args else kwargs["tx_hash"]
    status = None
    logger.info(f"⏳ Waiting for tx {get_tx_url(transaction_hash)}")
    while (
        status not in [TransactionStatus.ACCEPTED_ON_L2, TransactionStatus.REJECTED]
        and elapsed < max_wait
    ):
        if elapsed > 0:
            # don't log at the first iteration
            logger.info(f"ℹ️  Current status: {status}")
        logger.info(f"ℹ️  Sleeping for {check_interval}s")
        time.sleep(check_interval)
        response = requests.post(
            RPC_CLIENT.url,
            json={
                "jsonrpc": "2.0",
                "method": "starknet_getTransactionReceipt",
                "params": {"transaction_hash": hex(transaction_hash)},
                "id": 0,
            },
        )
        payload = json.loads(response.text)
        if payload.get("error"):
            if payload["error"]["message"] != "Transaction hash not found":
                logger.warn(json.dumps(payload["error"]))
                break
        status = payload.get("result", {}).get("status")
        if status is not None:
            status = TransactionStatus(status)
        else:
            # no status, but RPC currently doesn't return status for ACCEPTED_ON_L2 still PENDING
            # we take actual_fee as a proxy for ACCEPTED_ON_L2
            if payload.get("result", {}).get("actual_fee"):
                status = TransactionStatus.ACCEPTED_ON_L2
        elapsed = (datetime.now() - start).total_seconds()
    return status