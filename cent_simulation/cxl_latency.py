import math
import argparse

bandwidth_per_lane = 8
cxl_port_latency = 25
pcie_latency = 5+20+5
cxl_switch_latency = 20
one_hop_round_trip_latency = cxl_port_latency * 4 + pcie_latency * 2 + cxl_switch_latency
MULTICAST=True
if MULTICAST:
    pcie_latency *= 2
    bandwidth_per_lane *= 0.5


def llama_latency(size, PCIe_lanes_per_device, devices, total_devices):
    if devices <= total_devices:
        embedding_size = size[0]
        ffn_size = size[1]
        bandwidth_per_device = PCIe_lanes_per_device * bandwidth_per_lane
        if MULTICAST:
            multicast_param = 1
        else:
            multicast_param = total_devices // devices

        # 2B per value, 64B per message, 3 message per PBR flit
        embedding_broadcast_flits = math.ceil(embedding_size * 2 / 64 / 3) * multicast_param
        ffn_broadcast_flits = math.ceil(ffn_size * 2 / 64 / 3) * multicast_param
        embedding_broadcast_latency = (one_hop_round_trip_latency + embedding_broadcast_flits * 256 / bandwidth_per_device * 1000000000 / 1024 / 1024 / 1024) / 1000000
        ffn_broadcast_latency = (one_hop_round_trip_latency + ffn_broadcast_flits * 256 / bandwidth_per_device * 1000000000 / 1024 / 1024 / 1024) / 1000000

        embedding_gather_flits = math.ceil(embedding_size / devices * 2 / 64 / 3) * (devices - 1) * multicast_param
        ffn_gather_flits = math.ceil(ffn_size / devices * 2 / 64 / 3) * (devices - 1) * multicast_param
        embedding_gather_latency = (one_hop_round_trip_latency + embedding_gather_flits * 256 / bandwidth_per_device * 1000000000 / 1024 / 1024 / 1024) / 1000000
        ffn_gather_latency = (one_hop_round_trip_latency + ffn_gather_flits * 256 / bandwidth_per_device * 1000000000 / 1024 / 1024 / 1024) / 1000000

        total_latency = embedding_broadcast_latency * 4 + embedding_gather_latency * 5 + embedding_broadcast_latency * 1 + ffn_gather_latency  + ffn_broadcast_latency
        # print(total_latency)
        return total_latency

def gpt_latency(size, PCIe_lanes_per_device, devices, total_devices):
    if devices <= total_devices:
        embedding_size = size[0]
        ffn_size = size[1]
        bandwidth_per_device = PCIe_lanes_per_device * bandwidth_per_lane
        if MULTICAST:
            multicast_param = 1
        else:
            multicast_param = total_devices // devices

        embedding_broadcast_flits = math.ceil(embedding_size * 2 / 64 / 3) * multicast_param
        ffn_broadcast_flits = math.ceil(ffn_size * 2 / 64 / 3) * multicast_param
        embedding_broadcast_latency = (one_hop_round_trip_latency + embedding_broadcast_flits * 256 / bandwidth_per_device * 1000000000 / 1024 / 1024 / 1024) / 1000000
        ffn_broadcast_latency = (one_hop_round_trip_latency + ffn_broadcast_flits * 256 / bandwidth_per_device * 1000000000 / 1024 / 1024 / 1024) / 1000000

        embedding_gather_flits = math.ceil(embedding_size / devices * 2 / 64 / 3) * (devices - 1) * multicast_param
        ffn_gather_flits = math.ceil(ffn_size / devices * 2 / 64 / 3) * (devices - 1) * multicast_param
        embedding_gather_latency = (one_hop_round_trip_latency + embedding_gather_flits * 256 / bandwidth_per_device * 1000000000 / 1024 / 1024 / 1024) / 1000000
        ffn_gather_latency = (one_hop_round_trip_latency + ffn_gather_flits * 256 / bandwidth_per_device * 1000000000 / 1024 / 1024 / 1024) / 1000000

        total_latency = embedding_broadcast_latency * 4 + embedding_gather_latency * 5 + embedding_broadcast_latency * 1 +  ffn_gather_latency * 1
        print(total_latency)

def vector_latency(size, PCIe_lanes_per_device):
    bandwidth_per_device = PCIe_lanes_per_device * bandwidth_per_lane
    flits = math.ceil(size * 2 / 64 / 3)
    latency = (one_hop_round_trip_latency + flits * 256 / bandwidth_per_device * 1000000000 / 1024 / 1024 / 1024) / 1000000
    return latency

def vector_gather_latency(size, PCIe_lanes_per_device, devices, total_devices):
    if devices <= total_devices:
        embedding_size = size[0]
        bandwidth_per_device = PCIe_lanes_per_device * bandwidth_per_lane
        if MULTICAST:
            multicast_param = 1
        else:
            multicast_param = total_devices // devices

        # 2B per value, 64B per message, 3 message per PBR flit
        embedding_broadcast_flits = math.ceil(embedding_size * 2 / 64 / 3) * multicast_param
        embedding_broadcast_latency = (one_hop_round_trip_latency + embedding_broadcast_flits * 256 / bandwidth_per_device * 1000000000 / 1024 / 1024 / 1024) / 1000000

        embedding_gather_flits = math.ceil(embedding_size / devices * 2 / 64 / 3) * (devices - 1) * multicast_param
        embedding_gather_latency = (one_hop_round_trip_latency + embedding_gather_flits * 256 / bandwidth_per_device * 1000000000 / 1024 / 1024 / 1024) / 1000000

        total_latency = embedding_gather_latency
        print(total_latency)


def kv_head_tp_latency(hidden_dim, PCIe_lanes_per_device, devices, total_devices):
    """Latency in ms for the two row-parallel reductions in one block.

    Each reduction gathers one full hidden-dimension partial vector from every
    helper and broadcasts one full reduced vector.  The switch multicast model
    serializes a broadcast once, while the gather serializes ``devices - 1``
    helper vectors at the main-device ingress.
    """

    if devices <= 0 or total_devices <= 0:
        raise ValueError("device counts must be positive")
    if devices > total_devices:
        raise ValueError("TP devices cannot exceed total source devices")
    if hidden_dim <= 0 or PCIe_lanes_per_device <= 0:
        raise ValueError("hidden dimension and PCIe lanes must be positive")
    if devices == 1:
        return 0.0

    bandwidth_per_device = PCIe_lanes_per_device * bandwidth_per_lane
    vector_flits = math.ceil(hidden_dim * 2 / 64 / 3)
    gather_flits = vector_flits * (devices - 1)
    gather_latency = (
        one_hop_round_trip_latency
        + gather_flits * 256 / bandwidth_per_device * 1000000000 / 1024 / 1024 / 1024
    ) / 1000000
    broadcast_latency = (
        one_hop_round_trip_latency
        + vector_flits * 256 / bandwidth_per_device * 1000000000 / 1024 / 1024 / 1024
    ) / 1000000
    return 2.0 * (gather_latency + broadcast_latency)


def kv_head_tp_pcie_bits(hidden_dim, devices):
    """Return (main, per-helper, system) transmitted bits per block."""

    if hidden_dim <= 0 or devices <= 0:
        raise ValueError("hidden dimension and TP devices must be positive")
    if devices == 1:
        return 0, 0, 0
    # Two gathers are charged to the main and two broadcasts to each helper.
    # This partitions, without double counting, 4*(TP-1)*D BF16 link values.
    main_bits = 2 * (devices - 1) * hidden_dim * 16
    helper_bits = 2 * hidden_dim * 16
    system_bits = main_bits + (devices - 1) * helper_bits
    return main_bits, helper_bits, system_bits


def get_args():
    parser = argparse.ArgumentParser('Process model parameters.')
    parser.add_argument("--pipeline-parallel", action="store_true")
    parser.add_argument("--model-parallel", action="store_true")
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--model", choices=["Llama-7B", "Llama-13B", "Llama-70B", "GPT3-175B", "GPT3-175B-TP-8", "OPT-66B"], help="model choice")
    parser.add_argument("--num-devices", type=int, help="total devices")
    parser.add_argument("--group-devices", type=int, help="group devices in hybrid parallel")
    parser.add_argument("--PCIe-lanes", type=int, help="per device")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()

    embedding_size = {"Llama-7B": 4096, "Llama-13B": 5120, "Llama-70B": 8192, "GPT3-175B": 12288, "GPT3-175B-TP-8": 12288 // 8, "OPT-66B": 9216}
    ffn_size = {"Llama-7B": 11008, "Llama-13B": 13824, "Llama-70B": 28672, "GPT3-175B": 12288*4, "GPT3-175B-TP-8": 12288 // 8 * 4, "OPT-66B": 9216*4}

    if args.pipeline_parallel:
        print("pipeline_parallel", args.model, "{}_PCIe_lanes".format(args.PCIe_lanes), end=" ")
        latency = vector_latency(embedding_size[args.model], args.PCIe_lanes)
        print(latency)
    elif args.model_parallel:
        if args.embedding:
            print("model_parallel_embedding", args.model, "{}_devices".format(args.num_devices), "{}_group_devices".format(args.group_devices), "{}_PCIe_lanes".format(args.PCIe_lanes), end=" ")
            vector_gather_latency([embedding_size[args.model], ffn_size[args.model]], args.PCIe_lanes, args.group_devices, args.num_devices)
        else:
            if "Llama" in args.model:
                print("model_parallel", args.model, "{}_devices".format(args.num_devices), "{}_group_devices".format(args.group_devices), "{}_PCIe_lanes".format(args.PCIe_lanes), end=" ")
                latency = llama_latency([embedding_size[args.model], ffn_size[args.model]], args.PCIe_lanes, args.group_devices, args.num_devices)
                print(latency)
            else:
                print("model_parallel", args.model, "{}_devices".format(args.num_devices), "{}_group_devices".format(args.group_devices), "{}_PCIe_lanes".format(args.PCIe_lanes), end=" ")
                gpt_latency([embedding_size[args.model], ffn_size[args.model]], args.PCIe_lanes, args.group_devices, args.num_devices)






    # llama_devices = [2, 4, 8, 16]
    # bloom_devices = [3, 6, 16, 32]

    # vector_latency(54071*8, 4)
    # vector_latency(54071*8*16, 16)
    # vector_latency(4096, 4)
    # vector_latency(5120, 4)
    # vector_latency(8196, 4)
    # vector_latency(14336, 2)

    # for devices in llama_devices:
    #     print("llama devices:", devices)
    #     llama_latency([4096, 11008], 4, devices, 4)
    #     llama_latency([5120, 13824], 4, devices, 8)
    #     llama_latency([8196, 28672], 4, devices, 16)
    # for devices in bloom_devices:
    #     print("bloom devices:", devices)
    #     gpt_latency([14336, 14336*4], 2, devices, 32)
