"""
Compile DeepGEMM Kernels for a model with specify server arguments

This script launches a server for capturing DeepGEMM calls and then compiles the kernels.
It accepts server arguments (the same as launch_server.py).

Usage:
python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code

---

DeepGEMM 内核预编译工具

功能说明：
这是一个预编译工具，用于提前编译 DeepGEMM（FP8 量化 GEMM）内核，避免首次推理时的 JIT 编译延迟。

技术背景：
- DeepGEMM 使用 JIT 编译技术，针对特定矩阵形状 (M, N, K) 动态生成优化的 CUDA 内核
- 首次编译耗时：10-20 分钟
- 预编译后加载：仅需 1 秒

工作流程：
1. 设置环境变量，启用 DeepGEMM JIT 编译
2. 启动 sglang 服务器（禁用 CUDA Graph 和 Torch Compile 以节省时间）
3. 发送 dummy 推理请求触发所有 GEMM 操作
4. 遍历所有可能的 M 值（1 到 16K-128K），编译所有内核变体
5. 编译完成后关闭服务器并清理进程

适用场景：
- 主要用于 DeepSeek-V2/V3 等使用 FP8 量化和 MoE 架构的大模型
- 支持多 GPU (tensor parallel) 和多节点部署
- 编译结果缓存到磁盘，可复用

"""

import argparse
import dataclasses
import multiprocessing
import os
import time

import requests

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.srt.warmup import warmup

multiprocessing.set_start_method("spawn", force=True)

# ============================================================================
# 环境变量配置：控制 DeepGEMM 预编译行为
# ============================================================================

# 标记当前处于预编译阶段，减少不必要的警告信息
os.environ["SGL_IN_DEEPGEMM_PRECOMPILE_STAGE"] = "1"

# 强制启用 DeepGEMM JIT 编译（即使在默认情况下可能被禁用）
# DeepGEMM 是专门为 FP8 量化优化的 GEMM 库，提供比标准实现更高的性能
os.environ["SGL_ENABLE_JIT_DEEPGEMM"] = "1"

# 强制启用 MHA chunked KV 缓存，阈值设为 0
# 对于 DeepSeek V3 模型，这确保所有 kv_b_proj 相关的 DeepGEMM 内核都被编译
# 避免遗漏某些内核变体，导致运行时仍需编译
os.environ["SGL_CHUNKED_PREFIX_CACHE_THRESHOLD"] = "0"


@dataclasses.dataclass
class CompileArgs:
    """
    编译参数配置类

    Args:
        timeout: 编译超时时间（秒），默认 3600 秒（1 小时）
                 由于 DeepGEMM 需要编译大量内核变体（通常数千个），
                 整个过程可能需要 10-20 分钟或更长时间
    """
    timeout: int = 3600

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--timeout", type=int, default=CompileArgs.timeout)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


@warmup("compile-deep-gemm")
async def warm_up_compile(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """
    生成预热请求以触发 DeepGEMM 编译

    这个函数通过 @warmup 装饰器注册为 "compile-deep-gemm" 预热函数。
    当服务器启动时，如果设置了 warmups="compile-deep-gemm"，会自动调用此函数。

    工作原理：
    1. 创建一个简单的生成请求：4 个输入 token，生成 8 个输出 token
    2. 这个 dummy 请求会触发模型的前向传播，调用所有 GEMM 操作
    3. 第一次调用时，DeepGEMM 会为所有可能的矩阵形状编译内核
    4. 编译过程由 compile_utils.py 中的 _maybe_compile_deep_gemm_one_type_all 处理

    Args:
        disaggregation_mode: 分布式模式（null 或其他）
        tokenizer_manager: tokenizer 管理器实例
    """
    print("\nGenerate warm up request for compiling DeepGEMM...\n")
    generate_req_input = GenerateReqInput(
        input_ids=[0, 1, 2, 3],
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": 8,
            "ignore_eos": True,
        },
    )
    if disaggregation_mode != "null":
        generate_req_input.bootstrap_room = 0
        generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

    await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


def launch_server_internal(server_args):
    """
    在独立进程中启动服务器的内部函数

    这个函数会被多进程调用，确保在异常或正常退出时都能清理进程树。

    Args:
        server_args: 服务器参数配置
    """
    try:
        launch_server(server_args)
    except Exception as e:
        raise e
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def launch_server_process_and_send_one_request(
    server_args: ServerArgs, compile_args: CompileArgs
):
    """
    启动服务器进程并发送触发编译的请求

    多节点协调机制：
    - rank0 节点：启动服务器 → 等待就绪 → 发送 /generate 请求触发编译 → 返回
    - 非 rank0 节点：启动服务器 → 等待就绪 → 等待 rank0 节点完成 → 返回

    工作流程：
    1. 在独立进程中启动服务器
    2. 轮询健康检查接口，等待服务器就绪
    3. rank0 节点发送 dummy 推理请求，触发 DeepGEMM 编译
       - 编译过程会遍历所有可能的 M 值（1 到 m_max）
       - 为三种内核类型编译：普通 GEMM、连续分组 GEMM、带 mask 分组 GEMM
    4. 非 rank0 节点等待 rank0 节点完成

    Args:
        server_args: 服务器参数配置
        compile_args: 编译参数配置

    Returns:
        服务器进程对象

    Raises:
        TimeoutError: 如果在超时时间内服务器未就绪或编译未完成
        RuntimeError: 如果生成请求失败
    """
    proc = multiprocessing.Process(target=launch_server_internal, args=(server_args,))
    proc.start()
    base_url = f"http://{server_args.host}:{server_args.port}"
    timeout = compile_args.timeout

    start_time = time.perf_counter()
    while time.perf_counter() - start_time < timeout:
        try:
            headers = {
                "Content-Type": "application/json; charset=utf-8",
            }
            if server_args.node_rank == 0:
                response = requests.get(f"{base_url}/v1/models", headers=headers)
            else:
                # This http api is created by launch_dummy_health_check_server for none-rank0 node.
                response = requests.get(f"{base_url}/health", headers=headers)
            if response.status_code == 200:
                # Rank-0 node send a request to sync with other node and then return.
                if server_args.node_rank == 0:
                    response = requests.post(
                        f"{base_url}/generate",
                        json={
                            "input_ids": [0, 1, 2, 3],
                            "sampling_params": {
                                "max_new_tokens": 8,
                                "temperature": 0,
                            },
                        },
                        timeout=600,
                    )
                    if response.status_code != 200:
                        error = response.json()
                        raise RuntimeError(f"Sync request failed: {error}")
                # Other nodes should wait for the exit signal from Rank-0 node.
                else:
                    start_time_waiting = time.perf_counter()
                    while proc.is_alive():
                        if time.perf_counter() - start_time_waiting < timeout:
                            time.sleep(10)
                        else:
                            raise TimeoutError("Waiting for main node timeout!")
                return proc
        except requests.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError(
        "DeepGEMM Kernels compilation timeout."
        "\n\nFeel free and please restart the command."
    )


def refine_server_args(server_args: ServerArgs, compile_args: CompileArgs):
    """
    调整服务器参数以优化编译过程

    优化策略：
    1. 禁用 CUDA Graph：CUDA Graph 的初始化和捕获会增加启动时间，
       但在预编译阶段不需要这个优化，因此禁用以节省时间

    2. 禁用 Torch Compile：PyTorch 的 JIT 编译会增加额外的编译开销，
       在预编译阶段也不需要，因此禁用

    3. 延长 watchdog 超时：由于 DeepGEMM 编译可能需要很长时间（10-20 分钟），
       需要将 watchdog 超时设置为编译超时时间，避免被误杀

    4. 设置 warmup 函数：指定使用 "compile-deep-gemm" 预热函数，
       该函数会发送 dummy 请求触发编译

    Args:
        server_args: 服务器参数配置
        compile_args: 编译参数配置
    """
    # Disable cuda graph and torch compile to save time
    server_args.disable_cuda_graph = True
    server_args.enable_torch_compile = False
    print(f"Disable CUDA Graph and Torch Compile to save time...")

    # Set watchdog timeout to compile_args.timeout because compilation will take a long time
    server_args.watchdog_timeout = compile_args.timeout
    server_args.warmups = "compile-deep-gemm"


def run_compile(server_args: ServerArgs, compile_args: CompileArgs):
    """
    执行 DeepGEMM 内核编译的主函数

    编译内容：
    - 为三种内核类型编译（由 compile_utils.py 处理）：
      1. GEMM_NT_F8F8BF16：普通 FP8 GEMM（用于 Attention、Linear 等）
      2. GROUPED_GEMM_NT_F8F8BF16_CONTIG：连续布局的分组 GEMM（用于 MoE）
      3. GROUPED_GEMM_NT_F8F8BF16_MASKED：带 mask 的分组 GEMM（用于 MoE）

    - 每种类型都会为所有可能的 M 值编译（1 到 m_max）
      - m_max 取决于 chunked_prefill_size，通常是 16K-128K
      - 每个 M 值都会编译一个优化的内核

    编译结果：
    - 内核被缓存到 ~/.cache/deep_gemm/ 目录
    - 后续启动时只需加载缓存，仅需 1 秒左右

    Args:
        server_args: 服务器参数配置
        compile_args: 编译参数配置
    """
    print(
        "Begin DeepGEMM Kernels compilation...\n"
        "It may take a long time and timeout maybe raised "
        "while the compilation is still in progress.\n"
        "Just feel free to restart the command "
        "until the compilation is fully finished.\n"
    )

    proc = launch_server_process_and_send_one_request(server_args, compile_args)

    print("\nDeepGEMM Kernels compilation finished successfully.")

    # Sleep for safety
    time.sleep(10)
    if proc.is_alive():
        # This is the rank0 node.
        kill_process_tree(proc.pid)
    else:
        try:
            kill_process_tree(proc.pid)
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    CompileArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    compile_args = CompileArgs.from_cli_args(args)

    refine_server_args(server_args, compile_args)

    run_compile(server_args, compile_args)
