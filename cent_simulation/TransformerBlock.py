import math
import torch
import torch.nn.functional as F
from aim_sim import PIM
from utils import compare, apply_rotary_emb, repeat_kv, RMSNorm
from tp_mapping import KVHeadTPLayout, SystolicTPLayout, kv_head_tp_shape

debug = True

class TransformerBlock(PIM):
    """
    Llama TransformerBlock Class inherits computate functionality from PIM class
    """
    def __init__(self, dic_model, args):
        super().__init__(args)
        self.pim_compute = args.pim_compute
        if args.op_trace and not args.trace_score:
            self.trace_prepare = True
            self.trace_norm = True
            self.trace_fc_kqvo = True
            self.trace_attention = True
            self.trace_softmax = True
            self.trace_fc_kqvo = True
            self.trace_fc_ffn = True
            self.trace_activation = True
        else:
            self.trace_prepare = args.trace_prepare
            self.trace_norm = args.trace_norm
            self.trace_fc_kqvo = args.trace_fc_kqvo
            self.trace_attention = args.trace_attention
            self.trace_softmax = args.trace_softmax
            self.trace_fc_kqvo = args.trace_fc_kqvo
            self.trace_fc_ffn = args.trace_fc_ffn
            self.trace_activation = args.trace_activation
        self.trace_score = args.trace_score
        self.model = args.model
        self.seqlen = args.seqlen
        self.vocab_size = 32000
        self.FC_devices = args.FC_devices
        self.embedding = args.embedding
        self.only_FC = args.only_FC
        self.only_trace = args.only_trace
        self.model_parallel = args.model_parallel
        self.kv_head_tp = getattr(args, "kv_head_tp", False)
        self.tp_device_role = getattr(args, "tp_device_role", "main")
        self.pipeline_parallel = args.pipeline_parallel
        self.full_accelerator_softmax = args.full_accelerator_softmax
        self.flash_attention = args.flash_attention
        self.flash_attention_block_size = args.flash_attention_block_size
        # Match cent_dev: systolic PIM always moves element-wise multiplies to
        # the PNM vector multiplier, even when the CLI flag was not explicit.
        self.EWMUL_PNM = bool(
            getattr(args, "systolic_pim", False)
            or getattr(args, "EWMUL_PNM", False)
        )
        self.systolic_pim = args.systolic_pim
        self.systolic_dim = args.systolic_dim
        self.batch_size = args.batch_size
        self.microbench = args.microbench
        self.total_experts = args.total_experts
        self.active_experts = args.active_experts
        self.MLA = args.MLA
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.single_tb_per_device = args.single_tb_per_device
        if self.microbench and "SA" in self.microbench:
            self.systolic_dim = int(self.microbench[2:].split("x", 1)[0])
        if args.channels_per_block:
            self.channels_per_block = args.channels_per_block
        else:
            self.channels_per_block = args.num_channels
        self.GEMV_order = args.GEMV
        self.reuse_size = args.reuse_size
        if "TP_param" in dic_model.keys():
            self.TP_param = dic_model["TP_param"].item()
        else:
            self.TP_param = 1
        self.dim = dic_model["dim"].item()
        self.n_heads = dic_model["n_heads"].item()
        self.global_n_heads = (
            dic_model["global_n_heads"].item()
            if "global_n_heads" in dic_model
            else self.n_heads * self.TP_param
        )
        self.global_n_kv_heads = (
            dic_model["global_n_kv_heads"].item()
            if "global_n_kv_heads" in dic_model
            else self.n_heads
        )
        self.head_dim = (
            dic_model["head_dim"].item()
            if "head_dim" in dic_model
            else self.dim // self.n_heads // self.TP_param
        )
        self.max_seq_len = args.max_seq_len
        self.GQA = False
        self.inter_device_attention = args.inter_device_attention
        self.n_repeat = 1
        if "n_kv_heads" in dic_model.keys():
            self.GQA = True
            self.n_kv_heads = dic_model["n_kv_heads"].item()
            self.n_repeat = self.n_heads // self.n_kv_heads
        else:
            self.n_kv_heads = self.n_heads
        self.x = dic_model["x"].float()
        self.SANorm = dic_model["SANorm"].float()
        self.FFNNorm = dic_model["FFNNorm"].float()
        if "freqs_cis" in dic_model.keys():
            self.freqs_cis = dic_model["freqs_cis"]
        self.start_pos = dic_model["start_pos"]
        self.sa = dic_model["sa"].float()
        self.h = dic_model["h"].float()
        self.out = dic_model["out"].float()
        self.wq = dic_model["wq"].float()
        self.wk = dic_model["wk"].float()
        self.wv = dic_model["wv"].float()
        self.xq = dic_model["xq"].float()
        self.xk = dic_model["xk"].float()
        self.xv = dic_model["xv"].float()
        self.cache_k = dic_model["cache_k"].float()
        self.cache_v = dic_model["cache_v"].float()
        self.scores = dic_model["scores"].float()
        self.output = dic_model["output"].float()
        self.wo = dic_model["wo"].float()
        self.w1 = dic_model["w1"].float()
        self.w2 = dic_model["w2"].float()
        if "w3" in dic_model.keys():
            self.w3 = dic_model["w3"].float()
        self.ffn = dic_model["ffn"].float()
        self.kv_head_layout = None
        self.systolic_tp_layout = None
        if self.kv_head_tp:
            tp_shape = kv_head_tp_shape(
                dim=self.dim,
                query_heads=self.global_n_heads,
                kv_heads=self.global_n_kv_heads,
                ffn_dim=self.w1.shape[0] * self.FC_devices,
                tp=self.FC_devices,
                head_dim=self.head_dim,
            )
            self.kv_head_layout = KVHeadTPLayout(
                shape=tp_shape,
                num_channels=self.num_channels,
                banks_per_channel=self.num_banks,
                max_seq_len=self.max_seq_len,
                dram_columns=self.DRAM_column,
                burst_length=self.burst_length,
            )
            if self.systolic_pim:
                self.systolic_tp_layout = SystolicTPLayout(
                    shape=tp_shape,
                    num_channels=self.num_channels,
                    banks_per_channel=self.num_banks,
                    max_seq_len=self.max_seq_len,
                    systolic_height=self.systolic_dim,
                    dram_columns=self.DRAM_column,
                    burst_length=self.burst_length,
                )
        self.mode = {"vector":0, "weights":1, "cache_k":2, "cache_v":3, "score":4}
        self.total_banks = self.channels_per_block * self.num_banks
        if self.model_parallel:
            self.FC_total_banks = (
                self.total_banks if self.kv_head_tp else self.total_banks * self.FC_devices
            )
            self.intra_device_attention = (
                True if self.kv_head_tp else not self.inter_device_attention
            )
            banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1
            if banks_per_head < self.num_banks:
                self.intra_device_attention = True
        else:
            self.FC_total_banks = self.total_banks
            self.intra_device_attention = True

    def bank_index(self, index):
        # look for the bank to store a head
        dimm_index = index // (self.num_banks * self.num_channels)
        channel_index = (index - dimm_index * self.num_banks * self.num_channels) // self.num_banks
        bank_index = index % self.num_banks
        return dimm_index, channel_index, bank_index

    def store_to_DRAM_multi_channel_systolic_pim(self, data, row_index, mode, op_trace):
        if mode == self.mode["weights"]:
            data = data.T
            matrix_cols = data.shape[1]     # 1024: Wk in Llama3 70B
            matrix_rows = data.shape[0]     # 8192: Wk in Llama3 70B
            # The basic mapping idea is to first partition matrix_cols into banks, if matrix_cols_per_bank is smaller than burst_length, then we partition matrix_rows into banks to ensure each bank has at least one burst_length
            # matrix_cols = 8K, matrix_rows = 8K, 8 channels, 128 banks, matrix_cols_per_bank = 8K / 128 = 64, banks_per_matrix_col = 16 / 64 = 1, utilized_banks = 128, matrix_cols_per_bank = 64
            # matrix_cols = 1K, matrix_rows = 8K, 8 channels, 128 banks, matrix_cols_per_bank = 1K / 128 = 8, banks_per_matrix_col = 16 / 8 = 2, utilized_banks = 128, matrix_cols_per_bank = 16
            # matrix_cols = 1K, matrix_rows = 8K, 64 channels, 1K banks, matrix_cols_per_bank = 1K / 1K = 1, banks_per_matrix_col = 16 / 1 = 16 -> 8, utilized_banks = banks_per_matrix_col * matrix_cols / burst_length = 1K, matrix_cols_per_bank = 16
            matrix_cols_per_bank = math.ceil(matrix_cols / self.FC_total_banks)
            banks_per_matrix_col = math.ceil(self.burst_length / matrix_cols_per_bank)
            matrix_cols_per_bank = max(matrix_cols_per_bank, self.burst_length)
            bursts_of_matrix_cols_per_bank = math.ceil(matrix_cols_per_bank / self.burst_length)
            matrix_cols_per_bank_padding = bursts_of_matrix_cols_per_bank * self.burst_length  # round up to multiple of burst_length
            bursts_per_DRAM_row = self.DRAM_column // self.burst_length
            assert matrix_rows // banks_per_matrix_col * self.burst_length >= self.DRAM_column, "Each bank has more than one row of data"

            if banks_per_matrix_col > 1:    # Partition matrix_rows into banks
                banks_per_matrix_row = matrix_cols // self.burst_length
                banks_per_matrix_col = self.FC_total_banks // banks_per_matrix_row
                utilized_banks = banks_per_matrix_row * banks_per_matrix_col
                matrix_rows_per_bank = math.ceil(matrix_rows / banks_per_matrix_col)
                bursts_per_bank = math.ceil(matrix_rows_per_bank * matrix_cols_per_bank_padding / self.burst_length)
                DRAM_rows_per_bank = math.ceil(bursts_per_bank / bursts_per_DRAM_row)
                # print(utilized_banks, matrix_cols_per_bank_padding, matrix_rows_per_bank, banks_per_matrix_col, DRAM_rows_per_bank)
                for i in range(utilized_banks):
                    dimm_index, channel_index, bank_index = self.bank_index(i)
                    matrix_row_block_index = i // banks_per_matrix_row
                    matrix_col_block_index = i % banks_per_matrix_row
                    if i >= (banks_per_matrix_col-1) * banks_per_matrix_row:
                        matrix_rows_per_bank_per_row_block = matrix_rows - matrix_row_block_index * matrix_rows_per_bank
                    else:
                        matrix_rows_per_bank_per_row_block = matrix_rows_per_bank
                    data_block = data[matrix_row_block_index * matrix_rows_per_bank: matrix_row_block_index * matrix_rows_per_bank + matrix_rows_per_bank_per_row_block, matrix_col_block_index * matrix_cols_per_bank: (matrix_col_block_index + 1) * matrix_cols_per_bank]
                    shape = data_block.shape    # [matrix_rows_per_bank, matrix_cols_per_bank_padding]
                    for DRAM_row in range(DRAM_rows_per_bank):  # not considering fragmentation
                        if DRAM_row == DRAM_rows_per_bank - 1:
                            cols_load = matrix_rows_per_bank_per_row_block * self.burst_length - DRAM_row * self.DRAM_column
                        else:
                            cols_load = self.DRAM_column
                        data_row = data_block.reshape(-1)[DRAM_row * self.DRAM_column: DRAM_row * self.DRAM_column + cols_load]
                        self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + DRAM_row, 0, cols_load, data_row, op_trace)
            else:   # Keep all matrix_rows in a single bank, >=1 burst per bank
                utilized_banks = math.ceil(matrix_cols / matrix_cols_per_bank_padding)
                matrix_rows_per_bank = math.ceil(matrix_rows / banks_per_matrix_col)
                banks_per_matrix_row = utilized_banks
                bursts_per_bank = math.ceil(matrix_rows_per_bank * matrix_cols_per_bank_padding / self.burst_length)
                DRAM_rows_per_bank = math.ceil(bursts_per_bank / bursts_per_DRAM_row)
                DRAM_rows_per_burst_block = math.ceil(matrix_rows_per_bank / bursts_per_DRAM_row)
                # print(utilized_banks, matrix_cols_per_bank_padding, matrix_rows_per_bank, banks_per_matrix_col, DRAM_rows_per_burst_block)
                for i in range(utilized_banks):
                    dimm_index, channel_index, bank_index = self.bank_index(i)
                    if i == utilized_banks - 1:
                        matrix_cols_per_bank_per_col_block = matrix_cols - i * matrix_cols_per_bank_padding
                    else:
                        matrix_cols_per_bank_per_col_block = matrix_cols_per_bank_padding
                    data_block = data[: , i * matrix_cols_per_bank_padding: i * matrix_cols_per_bank_padding + matrix_cols_per_bank_per_col_block]

                    # Pad data_block_original if its columns are less than matrix_cols_per_bank_padding
                    if data_block.shape[1] < matrix_cols_per_bank_padding:
                        padding_needed = matrix_cols_per_bank_padding - data_block.shape[1]
                        data_block = F.pad(data_block, (0, padding_needed), "constant", 0)

                    shape = data_block.shape    # [matrix_rows_per_bank, matrix_cols_per_bank_padding]

                    for burst in range(bursts_of_matrix_cols_per_bank):
                        for DRAM_row in range(DRAM_rows_per_burst_block):  # not considering fragmentation
                            row_offset = burst * DRAM_rows_per_burst_block + DRAM_row
                            data_row = data_block[:, burst * self.burst_length: (burst+1) * self.burst_length].reshape(-1)[DRAM_row * bursts_per_DRAM_row * self.burst_length: DRAM_row * bursts_per_DRAM_row * self.burst_length + self.DRAM_column]
                            # if i == 0 and burst == 0 and DRAM_row == 1:
                            #     print(data_row)
                            self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + row_offset, 0, self.DRAM_column, data_row, op_trace)
        elif mode == self.mode["cache_k"]:
            # Store each seq in a bank
            seqlen = data.shape[1]
            shape = data.shape  # [8, 76, 32, 128]
            rows_per_head = self.head_dim * self.burst_length // self.DRAM_column   # 2
            total_bursts = math.ceil(seqlen / self.burst_length)
            bursts_per_bank = math.ceil(total_bursts / self.FC_total_banks)
            for batch_index in range(self.batch_size):
                batch_data = data[batch_index]
                for burst_index in range(total_bursts):
                    dimm_index, channel_index, bank_index = self.bank_index(burst_index % self.FC_total_banks)
                    for head in range(self.n_kv_heads):
                        if burst_index < total_bursts - 1:
                            data_per_burst_per_head = batch_data[burst_index * self.burst_length: (burst_index + 1) * self.burst_length, head].T.reshape(-1)   # [128, 16]
                        else:
                            data_slice = batch_data[burst_index * self.burst_length: seqlen, head]
                            data_per_burst_per_head = data_slice.T
                            if data_per_burst_per_head.shape[1] < self.burst_length:
                                padding_needed = self.burst_length - data_per_burst_per_head.shape[1]
                                data_per_burst_per_head = F.pad(
                                    data_per_burst_per_head,
                                    (0, padding_needed),
                                    "constant",
                                    0,
                                )
                            data_per_burst_per_head = data_per_burst_per_head.reshape(-1)
                        for row in range(rows_per_head):
                            row_offset = batch_index * self.n_kv_heads * rows_per_head * bursts_per_bank + self.n_kv_heads * rows_per_head * (burst_index // self.FC_total_banks) + head * rows_per_head + row
                            self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + row_offset, 0, self.DRAM_column, data_per_burst_per_head[row * self.DRAM_column: (row + 1) * self.DRAM_column], op_trace)
        elif mode == self.mode["cache_v"]:
            shape = data.shape  # [8, 76, 32, 128]
            # constant
            # each channel has 16 banks, each heads require 8 banks, each channel handle 2 heads
            num_banks_per_head_dim = self.head_dim // self.burst_length         # 8
            heads_per_channel = self.num_banks // num_banks_per_head_dim        # 2
            num_bursts_per_DRAM_row = self.DRAM_column // self.burst_length     # 64
            # Variable
            GQA = self.n_repeat
            systolic_dim_padding = min(self.systolic_dim, GQA)
            # systolic_pim = 8, chunk_size = 1024 // 8 = 128
            # systolic_pim = 1, chunk_size = 1024 // 8 = 1024
            seqlen = shape[1]
            chunk_size = self.DRAM_column // systolic_dim_padding
            chunks = (seqlen - 1) // chunk_size + 1
            # each channel has 16 banks, executing one KV head, if there num_channels < n_kv_heads, each channel has multiple KV heads, otherwise each channel has one KV head and chunks can be scaled down based on the number of channels
            pairs_of_kv_heads = self.n_kv_heads // heads_per_channel    # 4 in Llama-70B and 16 in Llama-7B
            if self.channels_per_block < pairs_of_kv_heads:
                paired_kv_head_per_channel = math.ceil(pairs_of_kv_heads / self.channels_per_block)
                seqlen_iterations = 1
            else:
                paired_kv_head_per_channel = 1
                seqlen_iterations = self.channels_per_block // pairs_of_kv_heads
            channels_per_seqlen_iteration = math.ceil(pairs_of_kv_heads / paired_kv_head_per_channel)
            channels_utilized = channels_per_seqlen_iteration * seqlen_iterations
            chunks_per_seqlen_iteration = math.ceil(chunks / seqlen_iterations)
            for batch_index in range(self.batch_size):
                batch_data = data[batch_index]
                for channel_index in range(channels_utilized):
                    dimm_index = channel_index // self.num_channels
                    for paired_kv_head in range(paired_kv_head_per_channel):
                        paired_kv_head_index = (channel_index * paired_kv_head_per_channel) % pairs_of_kv_heads + paired_kv_head
                        for seqlen_iteration in range(seqlen_iterations):
                            for chunk in range(chunks_per_seqlen_iteration):
                                chunk_index = seqlen_iteration * chunks_per_seqlen_iteration + chunk
                                # burst_index_per_head = bank_index_per_iteration % bursts_per_head
                                if chunk < chunks_per_seqlen_iteration - 1:
                                    seqlen_per_chunk = chunk_size
                                else:
                                    if chunk_index >= chunks:
                                        break
                                    seqlen_per_chunk = seqlen - (chunks_per_seqlen_iteration - 1) * chunk_size
                                for bank_index in range(self.num_banks):
                                    data_per_chunk_head_a = batch_data[chunk_index * chunk_size: chunk_index * chunk_size + seqlen_per_chunk, 2*paired_kv_head_index, bank_index * self.burst_length // 2: (bank_index + 1) * self.burst_length // 2]
                                    data_per_chunk_head_b = batch_data[chunk_index * chunk_size: chunk_index * chunk_size + seqlen_per_chunk, 2*paired_kv_head_index+1, bank_index * self.burst_length // 2: (bank_index + 1) * self.burst_length // 2]
                                    data_per_chunk = torch.cat((data_per_chunk_head_a, data_per_chunk_head_b), dim=1)
                                    rows_per_chunk = math.ceil(seqlen_per_chunk * self.burst_length / self.DRAM_column)
                                    for row in range(rows_per_chunk):
                                        if row == rows_per_chunk - 1:
                                            data_row = data_per_chunk.reshape(-1)[row * self.DRAM_column:]
                                        else:
                                            data_row = data_per_chunk.reshape(-1)[row * self.DRAM_column: (row + 1) * self.DRAM_column]
                                        row_len = data_row.shape[0]
                                        row_offset = batch_index * paired_kv_head_per_channel * seqlen_iterations * chunks_per_seqlen_iteration * rows_per_chunk + paired_kv_head * seqlen_iterations * chunks_per_seqlen_iteration * rows_per_chunk + seqlen_iteration * chunks_per_seqlen_iteration * rows_per_chunk + chunk * rows_per_chunk + row
                                        self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + row_offset, 0, row_len, data_row, op_trace)

        return shape

    def store_to_DRAM_multi_channel(self, data, row_index, mode, op_trace):
        if mode == self.mode["cache_k"]:
            # Store each seq in a bank
            seqlen = data.shape[0]
            shape = data.shape
            for seq in range(seqlen):
                dimm_index, channel_index, bank_index = self.bank_index(seq % self.FC_total_banks)
                data_seq = data[seq].reshape(-1)
                rows = (self.head_dim * self.n_kv_heads - 1) // self.DRAM_column + 1
                for row in range(rows):
                    data_row = data_seq[row * self.DRAM_column : (row + 1) * self.DRAM_column]
                    self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + seq // self.FC_total_banks * rows + row, 0, data_row.shape[0], data_row, op_trace)
                              
        elif mode == self.mode["cache_v"]:
            if self.intra_device_attention:
                seqlen = data.shape[-1]
                shape = data.shape
                rows_per_seq = (seqlen - 1) // self.DRAM_column + 1
                rows_per_dim = self.max_seq_len // self.DRAM_column
                num_heads_per_bank = (self.n_kv_heads - 1) // self.channels_per_block + 1
                dim_iterations = self.head_dim // self.num_banks
                for channel in range(self.channels_per_block):
                    if channel == self.channels_per_block - 1:
                        num_heads_iteration = self.n_kv_heads - num_heads_per_bank * (self.channels_per_block - 1)
                    else:
                        num_heads_iteration = num_heads_per_bank
                    for head_per_bank in range(num_heads_iteration):     # each head is distributed into all banks in a channel, each bank contains num_heads_per_bank heads
                        head = channel * num_heads_per_bank + head_per_bank
                        if head > self.n_kv_heads - 1:
                            break
                        row_current_head = row_index + (rows_per_dim * dim_iterations) * head_per_bank
                        for dim_iter in range(dim_iterations):   # each head has dim 128, but distributed to 16 banks, so has 8 iterations in each bank
                            for bank in range(self.num_banks):
                                dim = dim_iter * self.num_banks + bank
                                for row_offset in range(rows_per_seq):
                                    if row_offset == rows_per_seq - 1:
                                        self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_current_head + dim_iter * rows_per_dim + row_offset, 0, seqlen - self.DRAM_column * row_offset, data[head][dim][row_offset * self.DRAM_column:], op_trace)
                                    else:
                                        self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_current_head + dim_iter * rows_per_dim + row_offset, 0, self.DRAM_column, data[head][dim][row_offset * self.DRAM_column:(row_offset + 1) * self.DRAM_column], op_trace)
            else:
                seqlen = data.shape[-1]
                shape = data.shape
                channels_required_all_devices = self.FC_total_banks // self.num_banks
                # if banks_per_head < 16, channels_per_head < 1, one bank has more than one head, throw error
                # if 16 <= banks_per_head < 128, dim_iterations > 1, channels_per_head >= 1
                # if 128 <= banks_per_head < 512, dim_iterations = 1, devices_per_head = 1
                # if 512 <= banks_per_head, dim_iterations = 1, devices_per_head > 1
                                                                                                    # seqlen = 32k, head_dim = 128
                banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1                   # 32, 256, 2k
                channels_per_head = (banks_per_head - 1) // (self.num_banks) + 1                    # 2,  16,  128
                devices_per_head = (channels_per_head - 1) // (self.num_channels) + 1               # 1,  1,   4
                # iteration along the head dimension
                dim_iterations = (self.head_dim - 1) // banks_per_head + 1                          # 4,  1,   1
                # iteration along the sequence dimension or rows per sequence
                rows_per_seq_iteration = (banks_per_head - 1) // self.head_dim + 1                  # 1,  2,   16
                seq_iterations = (seqlen - 1) // (self.DRAM_column * rows_per_seq_iteration) + 1    # 32, 16,  2
                rows_per_seq = (seqlen - 1) // (self.DRAM_column) + 1                               # 32, 32,  32
                channels_per_row_offset = (self.head_dim - 1) // self.num_banks + 1                 # 8
                print("banks_per_head: ", banks_per_head)
                print("channels_per_head: ", channels_per_head)
                print("devices_per_head: ", devices_per_head)
                print("dim_iterations: ", dim_iterations)
                print("rows_per_seq_iteration: ", rows_per_seq_iteration)
                print("seq_iterations: ", seq_iterations)
                print("rows_per_seq: ", rows_per_seq)
                print("channels_per_row_offset: ", channels_per_row_offset)
                for channel in range(channels_required_all_devices):
                    if banks_per_head < self.num_banks:
                        raise ValueError("banks_per_head < self.num_banks. One head is mapped to less than one channel. Not enough channels are allocated.")
                    head = channel // (banks_per_head // self.num_banks)
                    if banks_per_head < 128:    # dim_iterations > 1, more than one dim in each row_offset are stored to a bank
                        for dim_iter in range(dim_iterations):   # E.g., head_dim = 128, banks_per_head = 32, channels_per_head = 2, dim_iterations = 128 / 32 = 4 in each bank. Within each iteration, Channel 0 is responsible for head 0: [0-15, 32-47, 64-79, 96-111], Channel 1 is responsible for head 1: [16-31, 48-63, 80-95, 112-127]. For bias vector, each head looks like (----CH0 16 Banks----,----CH1 16 Banks----) * 4.
                            for bank in range(self.num_banks):
                                dim = dim_iter * banks_per_head + (channel%channels_per_head) * self.num_banks + bank
                                for row_offset in range(rows_per_seq):
                                    if row_offset == rows_per_seq - 1:
                                        self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset * dim_iterations + dim_iter, 0, seqlen - self.DRAM_column * row_offset, data[head][dim][row_offset * self.DRAM_column:], op_trace)
                                    else:
                                        self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset * dim_iterations + dim_iter, 0, self.DRAM_column, data[head][dim][row_offset * self.DRAM_column:(row_offset + 1) * self.DRAM_column], op_trace)
                    else:
                        # each head is mapped on a single device, channels_per_row_offset = 128 / 16 = 8
                        # E.g., head_dim = 128, banks_per_head = 256, channels_per_head = 16. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 7 is responsible for head 0: [0][112-127], Channel 8 is responsible for head 0: [1][0-15], Channel 9 is responsible for head 0: [1][16-31], ..., Channel 15 is responsible for head 0: [1][112-127]. For bias vector, each head has rows_per_seq_iteration = 2: (----CH0 16 Banks----) * 8, (----CH8 16 Banks----) * 8.
                        # each head is mapped on multiple devices
                        # E.g. head_dim = 128, banks_per_head = 2048, channels_per_head = 128. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 127 is responsible for head 0: [15][112-127]. For bias vector, each head has rows_per_seq_iteration = 16: (----CH0 16 Banks----) * 128, ..., (----CH112 16 Banks----) * 128.
                        for bank in range(self.num_banks):
                            dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                            for row_offset in range(rows_per_seq):
                                if (channel % channels_per_head) // channels_per_row_offset == row_offset:
                                    if row_offset == rows_per_seq - 1:
                                        self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset // rows_per_seq_iteration, 0, seqlen - self.DRAM_column * row_offset, data[head][dim][row_offset * self.DRAM_column:], op_trace)
                                    else:
                                        self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset // rows_per_seq_iteration, 0, self.DRAM_column, data[head][dim][row_offset * self.DRAM_column:(row_offset + 1) * self.DRAM_column], op_trace)
        else:
            bank_dim = (data.shape[0] - 1) // self.total_banks + 1
            utilized_banks = (data.shape[0] - 1) // bank_dim + 1
            # print(data.shape, bank_dim, utilized_banks)
            if mode == self.mode["weights"]:
                if self.model_parallel:
                    bank_dim = (data.shape[0] - 1) // self.FC_total_banks + 1
                    utilized_banks = (data.shape[0] - 1) // bank_dim + 1
                for i in range(utilized_banks):
                    dimm_index, channel_index, bank_index = self.bank_index(i)
                    vector_length = data.shape[1]
                    rows_per_vector = (vector_length - 1) // self.DRAM_column + 1
                    if i < utilized_banks - 1:
                        num_vectors = bank_dim
                        shape = data[i * num_vectors : (i+1) * num_vectors].shape
                    else:
                        num_vectors = data.shape[0] - bank_dim * (utilized_banks - 1)
                    for vector in range(num_vectors):
                        data_vector = data[i * bank_dim + vector]
                        for row in range(rows_per_vector):
                            # print(i * bank_dim + vector, dimm_index, channel_index, bank_index, row_index + rows_per_vector * vector + row)
                            if row == rows_per_vector - 1:
                                data_tmp = data_vector[row * self.DRAM_column:]
                                self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + rows_per_vector * vector + row, 0, vector_length - row * self.DRAM_column, data_tmp, op_trace)
                            else:
                                data_tmp = data_vector[row * self.DRAM_column:(row + 1) * self.DRAM_column]
                                self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + rows_per_vector * vector + row, 0, self.DRAM_column, data_tmp, op_trace)
                # print(shape)
            elif mode == self.mode["vector"]:
                shape=data[:bank_dim].shape
                bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                for burst in range(bursts_per_bank):
                    for bank in range(utilized_banks):
                        dimm_index, channel_index, bank_index = self.bank_index(bank)
                        if bank < utilized_banks - 1:
                            if burst < bursts_per_bank - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: bank * bank_dim + (burst + 1) * self.burst_length]
                            else:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: (bank + 1) * bank_dim]
                        else:
                            last_bank_length = data.shape[0] - bank * bank_dim
                            last_bank_bursts = (last_bank_length - 1) // self.burst_length + 1
                            if burst < last_bank_bursts - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: bank * bank_dim + (burst + 1) * self.burst_length]
                            elif burst == last_bank_bursts - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length:]
                            else:
                                continue
                        self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, burst * self.burst_length, data_bank.shape[0], data_bank, op_trace)

                # for i in range(utilized_banks):
                #     dimm_index, channel_index, bank_index = self.bank_index(i)
                #     if i < utilized_banks - 1:
                #         data_bank = data[i * bank_dim : (i+1) * bank_dim]
                #         shape = data_bank.shape
                #     else:
                #         data_bank = data[i * bank_dim :]
                #     self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, 0, data_bank.shape[0], data_bank, op_trace)
            elif mode == self.mode["score"]:
                for i in range(self.total_banks):
                    dimm_index, channel_index, bank_index = self.bank_index(i)
                    data_bank = data[i].reshape(-1)
                    shape = data_bank.shape
                    data_size = data_bank.shape
                    rows = (data_size[0] - 1) // self.DRAM_column + 1
                    for row in range(rows-1):
                        data_tmp = data_bank[row * self.DRAM_column : (row + 1) * self.DRAM_column]
                        self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + row, 0, self.DRAM_column, data_tmp, op_trace)
                    data_tmp = data_bank[(rows-1) * self.DRAM_column:]
                    self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + rows - 1, 0, data_tmp.shape[0], data_tmp, op_trace)
            elif "vector_bank_group" in mode:
                # Gather the values in 4 banks in a bank group to 1 bank
                bank_dim = (data.shape[0] - 1) // (self.total_banks // 4) + 1
                utilized_banks = (data.shape[0] - 1) // bank_dim + 1

                shape=data[:bank_dim].shape
                bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                neighbor_bank_index = int(mode[-1])
                for burst in range(bursts_per_bank):
                    for bank in range(utilized_banks):
                        dimm_index, channel_index, bank_index = self.bank_index(bank*4+neighbor_bank_index)
                        if bank < utilized_banks - 1:
                            if burst < bursts_per_bank - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: bank * bank_dim + (burst + 1) * self.burst_length]
                            else:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: (bank + 1) * bank_dim]
                        else:
                            last_bank_length = data.shape[0] - bank * bank_dim
                            last_bank_bursts = (last_bank_length - 1) // self.burst_length + 1
                            if burst < last_bank_bursts - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: bank * bank_dim + (burst + 1) * self.burst_length]
                            elif burst == last_bank_bursts - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length:]
                            else:
                                continue
                        self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, burst * self.burst_length, data_bank.shape[0], data_bank, op_trace)

                # for i in range(utilized_banks):
                #     bank_group_index = int(mode[-1])
                #     dimm_index, channel_index, bank_index = self.bank_index(i*4+bank_group_index)
                #     if i < utilized_banks - 1:
                #         data_bank = data[i * bank_dim: (i+1) * bank_dim]
                #         shape = data_bank.shape
                #     else:
                #         data_bank = data[i * bank_dim:]
                #     self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, 0, data_bank.shape[0], data_bank, op_trace)
            elif "vector_neighbor_bank" in mode:
                # Gather the values in 2 neighboring banks in 1 bank
                bank_dim = (data.shape[0] - 1) // (self.total_banks // 2) + 1
                utilized_banks = (data.shape[0] - 1) // bank_dim + 1
                # print(data.shape, bank_dim, utilized_banks)
                shape=data[:bank_dim].shape
                bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                # e.g. Llama2-7B model has 4096 dim, with 10 channels, there are total 160 banks, bank_dim = 4096 // 80 + 1 = 52, bursts_per_bank = 52 // 16 + 1 = 4
                neighbor_bank_index = int(mode[-1])
                for burst in range(bursts_per_bank):
                    for bank in range(utilized_banks):
                        dimm_index, channel_index, bank_index = self.bank_index(bank*2+neighbor_bank_index)
                        if bank < utilized_banks - 1:
                            if burst < bursts_per_bank - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: bank * bank_dim + (burst + 1) * self.burst_length]
                            else:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: (bank + 1) * bank_dim]
                        else:
                            last_bank_length = data.shape[0] - bank * bank_dim
                            last_bank_bursts = (last_bank_length - 1) // self.burst_length + 1
                            if burst < last_bank_bursts - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: bank * bank_dim + (burst + 1) * self.burst_length]
                            elif burst == last_bank_bursts - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length:]
                            else:
                                continue
                        self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, burst * self.burst_length, data_bank.shape[0], data_bank, op_trace)

                # for i in range(utilized_banks):
                #     neighbor_bank_index = int(mode[-1])
                #     dimm_index, channel_index, bank_index = self.bank_index(i*2+neighbor_bank_index)
                #     if i < utilized_banks - 1:
                #         data_bank = data[i * bank_dim: (i+1) * bank_dim]
                #         shape = data_bank.shape
                #     else:
                #         data_bank = data[i * bank_dim:]
                #     self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, 0, data_bank.shape[0], data_bank, op_trace)
            elif "scores_bank_group" in mode:
                seqlen = data.shape[-1]
                # 4k scores use at most 4 rows
                # store each score in one bank, requring n_head banks
                rows_per_score = (seqlen - 1) // self.DRAM_column + 1
                num_heads_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
                shape = data.shape
                bank_group_index = int(mode[-1])
                for k in range(rows_per_score):
                    if k == rows_per_score - 1:
                        bank_dim = seqlen - self.DRAM_column * (rows_per_score - 1)
                    else:
                        bank_dim = self.DRAM_column
                    bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                    for j in range(num_heads_per_bank):
                        for burst in range(bursts_per_bank):
                            for bank in range(self.total_banks):
                                dimm_index, channel_index, bank_index = self.bank_index(bank)
                                if bank % 4 == bank_group_index:
                                    head = (bank // 4) * num_heads_per_bank + j
                                    if head > self.n_heads - 1:
                                        break
                                    if burst < bursts_per_bank - 1:
                                        data_bank = data[head][k * self.DRAM_column + burst * self.burst_length : k * self.DRAM_column + (burst + 1) * self.burst_length]
                                    else:
                                        data_bank = data[head][k * self.DRAM_column + burst * self.burst_length :]
                                    self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + j * rows_per_score + k, burst * self.burst_length, data_bank.shape[0], data_bank, op_trace)

                # for i in range(self.total_banks):
                #     dimm_index, channel_index, bank_index = self.bank_index(i)
                #     bank_group_index = int(mode[-1])
                #     shape = data.shape
                #     if i % 4 == bank_group_index:
                #         for j in range(num_heads_per_bank):
                #             head = (i // 4) * num_heads_per_bank + j
                #             if head > self.n_heads - 1:
                #                 break
                #             for k in range(rows_per_score):
                #                 if k == rows_per_score - 1:
                #                     data_bank = data[head][k * self.DRAM_column :]
                #                 else:
                #                     data_bank = data[head][k * self.DRAM_column : (k + 1) * self.DRAM_column]
                #                 self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + j * rows_per_score + k, 0, data_bank.shape[0], data_bank, op_trace)
        return shape
    
    def load_from_DRAM_multi_channel_systolic_pim(self, shape, row_index, mode, offset, op_trace):
        if mode == self.mode["weights"]:
            matrix_cols = shape[0]     # 1024: Wk in Llama3 70B
            matrix_rows = shape[1]     # 8192: Wk in Llama3 70B
            # The basic mapping idea is to first partition matrix_cols into banks, if matrix_cols_per_bank is smaller than burst_length, then we partition matrix_rows into banks to ensure each bank has at least one burst_length
            # matrix_cols = 8K, matrix_rows = 8K, 8 channels, 128 banks, matrix_cols_per_bank = 8K / 128 = 64, banks_per_matrix_col = 16 / 64 = 1, utilized_banks = 128, matrix_cols_per_bank = 64
            # matrix_cols = 1K, matrix_rows = 8K, 8 channels, 128 banks, matrix_cols_per_bank = 1K / 128 = 8, banks_per_matrix_col = 16 / 8 = 2, utilized_banks = 128, matrix_cols_per_bank = 16
            # matrix_cols = 1K, matrix_rows = 8K, 64 channels, 1K banks, matrix_cols_per_bank = 1K / 1K = 1, banks_per_matrix_col = 16 / 1 = 16 -> 8, utilized_banks = banks_per_matrix_col * matrix_cols / burst_length = 1K, matrix_cols_per_bank = 16
            matrix_cols_per_bank = math.ceil(matrix_cols / self.FC_total_banks)
            banks_per_matrix_col = math.ceil(self.burst_length / matrix_cols_per_bank)
            # print(matrix_cols, self.FC_total_banks, matrix_cols_per_bank, banks_per_matrix_col)
            matrix_cols_per_bank = max(matrix_cols_per_bank, self.burst_length)
            bursts_of_matrix_cols_per_bank = math.ceil(matrix_cols_per_bank / self.burst_length)
            matrix_cols_per_bank_padding = bursts_of_matrix_cols_per_bank * self.burst_length  # round up to multiple of burst_length
            bursts_per_DRAM_row = self.DRAM_column // self.burst_length
            assert matrix_rows // banks_per_matrix_col * self.burst_length >= self.DRAM_column, "Each bank has more than one row of data"
            result_tensor = torch.empty(matrix_rows, matrix_cols)

            if banks_per_matrix_col > 1:    # Partition matrix_rows into banks
                banks_per_matrix_row = matrix_cols // self.burst_length
                banks_per_matrix_col = self.FC_total_banks // banks_per_matrix_row
                utilized_banks = banks_per_matrix_row * banks_per_matrix_col
                matrix_rows_per_bank = math.ceil(matrix_rows / banks_per_matrix_col)
                bursts_per_bank = math.ceil(matrix_rows_per_bank * matrix_cols_per_bank_padding / self.burst_length)
                DRAM_rows_per_bank = math.ceil(bursts_per_bank / bursts_per_DRAM_row)
                # print(utilized_banks, matrix_cols_per_bank_padding, matrix_rows_per_bank, banks_per_matrix_col, DRAM_rows_per_bank)
                for i in range(utilized_banks):
                    dimm_index, channel_index, bank_index = self.bank_index(i)
                    matrix_row_block_index = i // banks_per_matrix_row
                    matrix_col_block_index = i % banks_per_matrix_row
                    if i >= (banks_per_matrix_col-1) * banks_per_matrix_row:
                        matrix_rows_per_bank_per_row_block = matrix_rows - matrix_row_block_index * matrix_rows_per_bank
                    else:
                        matrix_rows_per_bank_per_row_block = matrix_rows_per_bank
                    data_block = []
                    for DRAM_row in range(DRAM_rows_per_bank):  # not considering fragmentation
                        if DRAM_row == DRAM_rows_per_bank - 1:
                            cols_load = matrix_rows_per_bank_per_row_block * self.burst_length - DRAM_row * self.DRAM_column
                        else:
                            cols_load = self.DRAM_column
                        data_block.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + DRAM_row, 0, cols_load, op_trace))
                        # if i == 128 and DRAM_row == 0:
                        #     print("DRAM_row", DRAM_row, row_index + DRAM_row, data_block[DRAM_row].reshape(64, 16))
                    data_block_tensor = torch.cat(data_block).reshape(matrix_rows_per_bank_per_row_block, matrix_cols_per_bank_padding)
                    result_tensor[matrix_row_block_index * matrix_rows_per_bank: matrix_row_block_index * matrix_rows_per_bank + matrix_rows_per_bank_per_row_block, matrix_col_block_index * matrix_cols_per_bank: matrix_col_block_index * matrix_cols_per_bank + matrix_cols_per_bank] = data_block_tensor[:, :matrix_cols_per_bank]
            else:   # Keep all matrix_rows in a single bank, >=1 burst per bank
                utilized_banks = math.ceil(matrix_cols / matrix_cols_per_bank_padding)
                matrix_rows_per_bank = math.ceil(matrix_rows / banks_per_matrix_col)
                banks_per_matrix_row = utilized_banks
                bursts_per_bank = math.ceil(matrix_rows_per_bank * matrix_cols_per_bank_padding / self.burst_length)
                DRAM_rows_per_bank = math.ceil(bursts_per_bank / bursts_per_DRAM_row)
                DRAM_rows_per_burst_block = math.ceil(matrix_rows_per_bank / bursts_per_DRAM_row)
                # print(utilized_banks, bursts_of_matrix_cols_per_bank, matrix_cols_per_bank_padding, matrix_rows_per_bank, banks_per_matrix_col, DRAM_rows_per_burst_block)
                for i in range(utilized_banks):
                    dimm_index, channel_index, bank_index = self.bank_index(i)
                    if i == utilized_banks - 1:
                        matrix_cols_per_bank_per_col_block = matrix_cols - i * matrix_cols_per_bank_padding
                    else:
                        matrix_cols_per_bank_per_col_block = matrix_cols_per_bank_padding
                    for burst in range(bursts_of_matrix_cols_per_bank):
                        data_block = []
                        for DRAM_row in range(DRAM_rows_per_burst_block):  # not considering fragmentation
                            row_offset = burst * DRAM_rows_per_burst_block + DRAM_row
                            data_block.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + row_offset, 0, self.DRAM_column, op_trace))
                            # if i == 0 and burst == 0 and DRAM_row == 1:
                            #     print(data_block[0])
                        data_block_tensor = torch.cat(data_block).reshape(matrix_rows_per_bank, self.burst_length)
                        if i < utilized_banks // banks_per_matrix_col - 1:
                            result_tensor[:, i * matrix_cols_per_bank_padding + burst * self.burst_length: i * matrix_cols_per_bank_padding + (burst+1) * self.burst_length] = data_block_tensor
                        else:
                            matrix_cols_per_bank_per_col_block = matrix_cols - i * matrix_cols_per_bank_padding
                            if burst * self.burst_length < matrix_cols_per_bank_per_col_block:
                                result_tensor[:, i * matrix_cols_per_bank_padding + burst * self.burst_length: i * matrix_cols_per_bank_padding + (burst+1) * self.burst_length] = data_block_tensor
                            else:
                                cols_load = matrix_cols - i * matrix_cols_per_bank_padding - burst * self.burst_length
                                result_tensor[:, i * matrix_cols_per_bank_padding: i * matrix_cols_per_bank_padding + cols_load] = data_block_tensor[:, :cols_load]
            result_tensor = result_tensor.T
        elif mode == self.mode["cache_k"]:
            seqlen = shape[1]   # [8, 76, 32, 128]
            rows_per_head = self.head_dim * self.burst_length // self.DRAM_column   # 2
            total_bursts = math.ceil(seqlen / self.burst_length)
            bursts_per_bank = math.ceil(total_bursts / self.FC_total_banks)
            result_tensor = torch.empty(shape[0], shape[1], shape[2], shape[3])
            for i in range(self.batch_size):
                batch_tensor = torch.empty(shape[2], shape[1], shape[3])
                for burst_index in range(total_bursts):
                    dimm_index, channel_index, bank_index = self.bank_index(burst_index % self.FC_total_banks)
                    for head in range(self.n_kv_heads):
                        data_load = []
                        for row in range(rows_per_head):
                            row_offset = i * self.n_kv_heads * rows_per_head * bursts_per_bank + self.n_kv_heads * rows_per_head * (burst_index // self.FC_total_banks) + head * rows_per_head + row
                            data_load.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + row_offset, 0, self.DRAM_column, op_trace))
                        data_load_tensor = torch.cat(data_load).reshape(self.head_dim, self.burst_length)
                        if burst_index < total_bursts - 1:
                            batch_tensor[head][burst_index * self.burst_length: (burst_index + 1) * self.burst_length] = data_load_tensor.T
                        else:
                            batch_tensor[head][burst_index * self.burst_length:] = data_load_tensor[:, :seqlen - (total_bursts - 1) * self.burst_length].T
                result_tensor[i] = batch_tensor.transpose(0, 1)
        elif mode == self.mode["cache_v"]:
            # shape = data.shape  # [8, 76, 32, 128]
            # constant
            # each channel has 16 banks, each heads require 8 banks, each channel handle 2 heads
            num_banks_per_head_dim = self.head_dim // self.burst_length         # 8
            heads_per_channel = self.num_banks // num_banks_per_head_dim        # 2
            num_bursts_per_DRAM_row = self.DRAM_column // self.burst_length     # 64
            # Variable
            GQA = self.n_repeat
            systolic_dim_padding = min(self.systolic_dim, GQA)
            # systolic_pim = 8, chunk_size = 1024 // 8 = 128
            # systolic_pim = 1, chunk_size = 1024 // 8 = 1024
            seqlen = shape[1]
            chunk_size = self.DRAM_column // systolic_dim_padding
            chunks = (seqlen - 1) // chunk_size + 1
            # each channel has 16 banks, executing one KV head, if there num_channels < n_kv_heads, each channel has multiple KV heads, otherwise each channel has one KV head and chunks can be scaled down based on the number of channels
            pairs_of_kv_heads = self.n_kv_heads // heads_per_channel    # 4 in Llama-70B and 16 in Llama-7B
            if self.channels_per_block < pairs_of_kv_heads:
                paired_kv_head_per_channel = math.ceil(pairs_of_kv_heads / self.channels_per_block)
                seqlen_iterations = 1
            else:
                paired_kv_head_per_channel = 1
                seqlen_iterations = self.channels_per_block // pairs_of_kv_heads
            channels_per_seqlen_iteration = math.ceil(pairs_of_kv_heads / paired_kv_head_per_channel)
            channels_utilized = channels_per_seqlen_iteration * seqlen_iterations
            chunks_per_seqlen_iteration = math.ceil(chunks / seqlen_iterations)
            # print("pairs_of_kv_heads: ", pairs_of_kv_heads, "paired_kv_head_per_channel: ", paired_kv_head_per_channel, "seqlen_iterations: ", seqlen_iterations, "channels_per_seqlen_iteration: ", channels_per_seqlen_iteration, "chunks_per_seqlen_iteration: ", chunks_per_seqlen_iteration, "channels_utilized: ", channels_utilized)

            result_tensor = torch.empty(shape[0], shape[1], shape[2], shape[3])
            for batch_index in range(self.batch_size):
                batch_tensor = torch.empty(shape[2], shape[1], shape[3])
                for channel_index in range(channels_utilized):
                    dimm_index = channel_index // self.num_channels
                    for paired_kv_head in range(paired_kv_head_per_channel):
                        paired_kv_head_index = (channel_index * paired_kv_head_per_channel) % pairs_of_kv_heads + paired_kv_head
                        for seqlen_iteration in range(seqlen_iterations):
                            for chunk in range(chunks_per_seqlen_iteration):
                                chunk_index = seqlen_iteration * chunks_per_seqlen_iteration + chunk
                                # burst_index_per_head = bank_index_per_iteration % bursts_per_head
                                if chunk < chunks_per_seqlen_iteration - 1:
                                    seqlen_per_chunk = chunk_size
                                else:
                                    if chunk_index >= chunks:
                                        break
                                    seqlen_per_chunk = seqlen - (chunks_per_seqlen_iteration - 1) * chunk_size
                                for bank_index in range(self.num_banks):
                                    data_load = []
                                    rows_per_chunk = math.ceil(seqlen_per_chunk * self.burst_length / self.DRAM_column)
                                    for row in range(rows_per_chunk):
                                        if row == rows_per_chunk - 1:
                                            row_len = seqlen_per_chunk * self.burst_length - row * self.DRAM_column
                                        else:
                                            row_len = self.DRAM_column
                                        row_offset = batch_index * paired_kv_head_per_channel * seqlen_iterations * chunks_per_seqlen_iteration * rows_per_chunk + paired_kv_head * seqlen_iterations * chunks_per_seqlen_iteration * rows_per_chunk + seqlen_iteration * chunks_per_seqlen_iteration * rows_per_chunk + chunk * rows_per_chunk + row
                                        data_load.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + row_offset, 0, row_len, op_trace))
                                    data_load_tensor = torch.cat(data_load).reshape(seqlen_per_chunk, self.burst_length)
                                    data_per_chunk_head_a = data_load_tensor[:, :self.burst_length // 2]
                                    data_per_chunk_head_b = data_load_tensor[:, self.burst_length // 2:]
                                    batch_tensor[2*paired_kv_head_index][chunk_index * chunk_size: chunk_index * chunk_size + seqlen_per_chunk, bank_index * self.burst_length // 2: (bank_index + 1) * self.burst_length // 2] = data_per_chunk_head_a
                                    batch_tensor[2*paired_kv_head_index + 1][chunk_index * chunk_size: chunk_index * chunk_size + seqlen_per_chunk, bank_index * self.burst_length // 2: (bank_index + 1) * self.burst_length // 2] = data_per_chunk_head_b
                                    # if paired_kv_head_index == pairs_of_kv_heads - 1 and bank_index == self.num_banks - 1:
                                    #     print(batch_tensor.shape)
                                    #     print(batch_tensor)
                result_tensor[batch_index] = batch_tensor.transpose(0, 1)

        return result_tensor

    def load_from_DRAM_multi_channel(self, shape, row_index, mode, offset, op_trace):
        result = []
        if mode == self.mode["cache_k"]:
            seqlen = offset
            for seq in range(seqlen):
                seqs = []
                dimm_index, channel_index, bank_index = self.bank_index(seq % self.FC_total_banks)
                rows = (self.head_dim * self.n_kv_heads - 1) // self.DRAM_column + 1
                for row in range(rows):
                    row_elements = min(
                        self.DRAM_column,
                        self.head_dim * self.n_kv_heads - row * self.DRAM_column,
                    )
                    seqs.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + seq // self.FC_total_banks * rows + row, 0, row_elements, op_trace))
                result.append(torch.cat(seqs).reshape(-1))
        elif mode == self.mode["cache_v"]:
            if self.intra_device_attention:
                seqlen = offset
                rows_per_seq = (seqlen - 1) // self.DRAM_column + 1
                rows_per_dim = self.max_seq_len // self.DRAM_column
                num_heads_per_bank = (self.n_kv_heads - 1) // self.channels_per_block + 1
                for channel in range(self.channels_per_block):
                    if channel == self.channels_per_block - 1:
                        num_heads_iteration = self.n_kv_heads - num_heads_per_bank * (self.channels_per_block - 1)
                    else:
                        num_heads_iteration = num_heads_per_bank
                    dim_iterations = self.head_dim // self.num_banks
                    for head_per_bank in range(num_heads_iteration):     # each head is distributed into all banks in a channel, each bank contains num_heads_per_bank heads
                        result_head = []
                        head = channel * num_heads_per_bank + head_per_bank
                        if head > self.n_kv_heads - 1:
                            break
                        row_current_head = row_index + (rows_per_dim * dim_iterations) * head_per_bank
                        for dim_iter in range(dim_iterations):   # each head has dim 128, but distributed to 16 banks, so has 8 iterations in each bank
                            for bank in range(self.num_banks):
                                result_dim = []
                                for row_offset in range(rows_per_seq):
                                    if row_offset == rows_per_seq - 1:
                                        result_dim.append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_current_head + dim_iter * rows_per_dim + row_offset, 0, seqlen - self.DRAM_column * row_offset, op_trace))
                                    else:
                                        result_dim.append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_current_head + dim_iter * rows_per_dim + row_offset, 0, self.DRAM_column, op_trace))
                                result_head.append(torch.cat(result_dim))
                        result.append(torch.cat(result_head))
            else:
                seqlen = offset
                channels_required_all_devices = self.FC_total_banks // self.num_banks
                # if banks_per_head < 16, channels_per_head < 1, one bank has more than one head, throw error
                # if 16 <= banks_per_head < 128, dim_iterations > 1, channels_per_head >= 1
                # if 128 <= banks_per_head < 512, dim_iterations = 1, devices_per_head = 1
                # if 512 <= banks_per_head, dim_iterations = 1, devices_per_head > 1
                                                                                                    # seqlen = 32k, head_dim = 128
                banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1                   # 32, 256, 2k
                channels_per_head = (banks_per_head - 1) // (self.num_banks) + 1                    # 2,  16,  128
                devices_per_head = (channels_per_head - 1) // (self.num_channels) + 1               # 1,  1,   4
                # iteration along the head dimension
                dim_iterations = (self.head_dim - 1) // banks_per_head + 1                          # 4,  1,   1
                # iteration along the sequence dimension or rows per sequence
                rows_per_seq_iteration = (banks_per_head - 1) // self.head_dim + 1                  # 1,  2,   16
                seq_iterations = (seqlen - 1) // (self.DRAM_column * rows_per_seq_iteration) + 1    # 32, 16,  2
                rows_per_seq = (seqlen - 1) // (self.DRAM_column) + 1                               # 32, 32,  32
                channels_per_row_offset = (self.head_dim - 1) // self.num_banks + 1                 # 8
                result_heads = [[[] for dim in range(self.head_dim)] for head in range(self.n_kv_heads)]
                for channel in range(channels_required_all_devices):
                    if banks_per_head < self.num_banks:
                        raise ValueError("banks_per_head < self.num_banks. One head is mapped to less than one channel. Not enough channels are allocated.")
                    head = channel // (banks_per_head // self.num_banks)
                    if banks_per_head < 128:    # dim_iterations > 1, more than one dim in each row_offset are stored to a bank
                        for dim_iter in range(dim_iterations):   # E.g., head_dim = 128, banks_per_head = 32, channels_per_head = 2, dim_iterations = 128 / 32 = 4 in each bank. Within each iteration, Channel 0 is responsible for head 0: [0-15, 32-47, 64-79, 96-111], Channel 1 is responsible for head 1: [16-31, 48-63, 80-95, 112-127]. For bias vector, each head looks like (----CH0 16 Banks----,----CH1 16 Banks----) * 4.
                            for bank in range(self.num_banks):
                                dim = dim_iter * banks_per_head + (channel%channels_per_head) * self.num_banks + bank
                                for row_offset in range(rows_per_seq):
                                    if row_offset == rows_per_seq - 1:
                                        result_heads[head][dim].append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset * dim_iterations + dim_iter, 0, seqlen - self.DRAM_column * row_offset, op_trace))
                                    else:
                                        result_heads[head][dim].append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset * dim_iterations + dim_iter, 0, self.DRAM_column, op_trace))
                    else:
                        # each head is mapped on a single device, channels_per_row_offset = 128 / 16 = 8
                        # E.g., head_dim = 128, banks_per_head = 256, channels_per_head = 16. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 7 is responsible for head 0: [0][112-127], Channel 8 is responsible for head 0: [1][0-15], Channel 9 is responsible for head 0: [1][16-31], ..., Channel 15 is responsible for head 0: [1][112-127]. For bias vector, each head has rows_per_seq_iteration = 2: (----CH0 16 Banks----) * 8, (----CH8 16 Banks----) * 8.
                        # each head is mapped on multiple devices
                        # E.g. head_dim = 128, banks_per_head = 2048, channels_per_head = 128. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 127 is responsible for head 0: [15][112-127]. For bias vector, each head has rows_per_seq_iteration = 16: (----CH0 16 Banks----) * 128, ..., (----CH112 16 Banks----) * 128.
                        for bank in range(self.num_banks):
                            dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                            for row_offset in range(rows_per_seq):
                                if (channel % channels_per_head) // channels_per_row_offset == row_offset:
                                    if row_offset == rows_per_seq - 1:
                                        result_heads[head][dim].append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset // rows_per_seq_iteration, 0, seqlen - self.DRAM_column * row_offset, op_trace))
                                    else:
                                        result_heads[head][dim].append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset // rows_per_seq_iteration, 0, self.DRAM_column, op_trace))
                for head in range(self.n_kv_heads):
                    result_head = []
                    for dim in range(self.head_dim):
                        result_head.append(torch.cat(result_heads[head][dim]))
                    result.append(torch.cat(result_head))
        else:
            # print(shape, offset)
            if mode == self.mode["weights"]:
                vector_length = shape[1]
                rows_per_vector = (vector_length - 1) // self.DRAM_column + 1
                utilized_banks = (shape[0] - 1) // offset + 1  # shape = [4096, 11008]
                for i in range(utilized_banks):
                    dimm_index, channel_index, bank_index = self.bank_index(i)
                    rows_per_vector = (vector_length - 1) // self.DRAM_column + 1
                    vectors = []
                    # for vector in range(self.head_dim):
                    if i < utilized_banks - 1:
                        num_vectors = offset
                    else:
                        num_vectors = shape[0] - offset * (utilized_banks - 1)
                    for vector in range(num_vectors):
                        rows = []
                        for row in range(rows_per_vector):
                            if row == rows_per_vector - 1:
                                rows.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + rows_per_vector * vector + row, 0, vector_length - row * self.DRAM_column, op_trace))
                            else:
                                rows.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + rows_per_vector * vector + row, 0, self.DRAM_column, op_trace))
                        vectors.append(torch.cat(rows))
                    result.append(torch.cat(vectors))
                # print(torch.cat(result).reshape(shape).shape)
            elif mode == self.mode["vector"]:
                utilized_banks = (shape[-1] - 1) // offset + 1  # shape = [1, 1, 4096]
                bank_dim = offset
                bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                result_banks = [[] for _ in range(utilized_banks)]
                for burst in range(bursts_per_bank):
                    for bank in range(utilized_banks):
                        dimm_index, channel_index, bank_index = self.bank_index(bank)
                        if bank < utilized_banks - 1:
                            if burst < bursts_per_bank - 1:
                                num_cols = self.burst_length
                            else:
                                num_cols = bank_dim - burst * self.burst_length
                        else:
                            last_bank_length = shape[-1] - bank * bank_dim
                            last_bank_bursts = (last_bank_length - 1) // self.burst_length + 1
                            if burst < last_bank_bursts - 1:
                                num_cols = self.burst_length
                            elif burst == last_bank_bursts - 1:
                                num_cols = last_bank_length - burst * self.burst_length
                            else:
                                continue
                        result_banks[bank].append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, burst * self.burst_length, num_cols, op_trace))
                for bank in range(utilized_banks):
                    result += result_banks[bank]

                # for i in range(utilized_banks):
                #     dimm_index, channel_index, bank_index = self.bank_index(i)
                #     if i < utilized_banks - 1:
                #         num_cols = offset
                #     else:
                #         num_cols = shape[-1] - offset * (utilized_banks - 1)
                #     result.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, 0, num_cols, op_trace))
            elif mode == self.mode["score"]:
                for i in range(self.total_banks):
                    dimm_index, channel_index, bank_index = self.bank_index(i)
                    rows = []
                    rows_used = (offset - 1) // self.DRAM_column + 1
                    for row in range(rows_used-1):
                        rows.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + row, 0, self.DRAM_column, op_trace))
                    rows.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + rows_used - 1, 0, offset - (rows_used - 1) * self.DRAM_column, op_trace))
                    result.append(torch.cat(rows).reshape(-1))
            elif "vector_bank_group" in mode:
                utilized_banks = (shape[-1] - 1) // offset + 1  # shape = [1, 1, 4096]
                # print(utilized_banks)
                bank_dim = offset
                bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                result_banks = [[] for _ in range(utilized_banks)]
                for burst in range(bursts_per_bank):
                    for bank in range(utilized_banks):
                        neighbor_bank_index = int(mode[-1])
                        dimm_index, channel_index, bank_index = self.bank_index(bank*4+neighbor_bank_index)
                        if bank < utilized_banks - 1:
                            if burst < bursts_per_bank - 1:
                                num_cols = self.burst_length
                            else:
                                num_cols = bank_dim - burst * self.burst_length
                        else:
                            last_bank_length = shape[-1] - bank * bank_dim
                            last_bank_bursts = (last_bank_length - 1) // self.burst_length + 1
                            if burst < last_bank_bursts - 1:
                                num_cols = self.burst_length
                            elif burst == last_bank_bursts - 1:
                                num_cols = last_bank_length - burst * self.burst_length
                            else:
                                continue
                        result_banks[bank].append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, burst * self.burst_length, num_cols, op_trace))
                for bank in range(utilized_banks):
                    result += result_banks[bank]

                # for i in range(utilized_banks):
                #     neighbor_bank_index = int(mode[-1])
                #     dimm_index, channel_index, bank_index = self.bank_index(i*4+neighbor_bank_index)
                #     if i < utilized_banks - 1:
                #         num_cols = offset
                #     else:
                #         num_cols = shape[-1] - offset * (utilized_banks - 1)
                #     result.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, 0, num_cols, op_trace))
            elif "vector_neighbor_bank" in mode:
                utilized_banks = (shape[-1] - 1) // offset + 1  # shape = [1, 1, 4096]
                bank_dim = offset
                bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                result_banks = [[] for _ in range(utilized_banks)]
                for burst in range(bursts_per_bank):
                    for bank in range(utilized_banks):
                        neighbor_bank_index = int(mode[-1])
                        dimm_index, channel_index, bank_index = self.bank_index(bank*2+neighbor_bank_index)
                        if bank < utilized_banks - 1:
                            if burst < bursts_per_bank - 1:
                                num_cols = self.burst_length
                            else:
                                num_cols = bank_dim - burst * self.burst_length
                        else:
                            last_bank_length = shape[-1] - bank * bank_dim
                            last_bank_bursts = (last_bank_length - 1) // self.burst_length + 1
                            if burst < last_bank_bursts - 1:
                                num_cols = self.burst_length
                            elif burst == last_bank_bursts - 1:
                                num_cols = last_bank_length - burst * self.burst_length
                            else:
                                continue
                        result_banks[bank].append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, burst * self.burst_length, num_cols, op_trace))
                for bank in range(utilized_banks):
                    result += result_banks[bank]

                # for i in range(utilized_banks):
                #     neighbor_bank_index = int(mode[-1])
                #     dimm_index, channel_index, bank_index = self.bank_index(i*2+neighbor_bank_index)
                #     if i < utilized_banks - 1:
                #         num_cols = offset
                #     else:
                #         num_cols = shape[-1] - offset * (utilized_banks - 1)
                #     result.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, 0, num_cols, op_trace))
            elif "scores_bank_group" in mode:
                seqlen = offset
                # 4k scores use at most 4 rows
                # store each score in one bank, requring n_head banks
                rows_per_score = (seqlen - 1) // self.DRAM_column + 1
                num_heads_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
                bank_group_index = int(mode[-1])
                score_heads = [[] for _ in range(self.n_heads)]
                for k in range(rows_per_score):
                    if k == rows_per_score - 1:
                        bank_dim = seqlen - self.DRAM_column * (rows_per_score - 1)
                    else:
                        bank_dim = self.DRAM_column
                    bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                    for j in range(num_heads_per_bank):
                        for burst in range(bursts_per_bank):
                            for bank in range(self.total_banks):
                                dimm_index, channel_index, bank_index = self.bank_index(bank)
                                if bank % 4 == bank_group_index:
                                    head = (bank // 4) * num_heads_per_bank + j
                                    if head > self.n_heads - 1:
                                        break
                                    if burst < bursts_per_bank - 1:
                                        num_cols = self.burst_length
                                    else:
                                        num_cols = bank_dim - burst * self.burst_length
                                    score_heads[head].append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + j * rows_per_score + k, burst * self.burst_length, num_cols, op_trace))
                for head in range(self.n_heads):
                    score = torch.cat(score_heads[head])
                    result.append(score)

                # for i in range(self.total_banks):
                #     dimm_index, channel_index, bank_index = self.bank_index(i)
                #     bank_group_index = int(mode[-1])
                #     if i % 4 == bank_group_index:
                #         for j in range(num_heads_per_bank):
                #             head = (i // 4) * num_heads_per_bank + j
                #             if head > self.n_heads - 1:
                #                 break
                #             scores = []
                #             for k in range(rows_per_score):
                #                 if k == rows_per_score - 1:
                #                     scores.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + j * rows_per_score + k, 0, offset - k * self.DRAM_column, op_trace))
                #                 else:
                #                     scores.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + j * rows_per_score + k, 0, self.DRAM_column, op_trace))
                #             result.append(torch.cat(scores))
        return torch.cat(result).reshape(shape)

    def broadcast_store_query(self, channels_required, dest_row_index, data, op_trace):
        # match GQA score with key cache
        # xq[0][128] xq[8][128]  ... xq[56][128]
        # xq[1][128] xq[9][128]  ... xq[57][128]
        #                        ...
        # xq[7][128] xq[15][128] ... xq[63][128]
        for dest_channel in range(channels_required):
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + data.shape[-1] // self.burst_length
            for bank in range(self.dim//self.DRAM_column):
                if self.GQA:
                    data_tmp = torch.cat([data[i * self.DRAM_column + bank * self.head_dim : i * self.DRAM_column + (bank + 1) * self.head_dim] for i in range(self.n_kv_heads)])
                else:
                    data_tmp = data[bank * self.DRAM_column : (bank + 1) * self.DRAM_column]
                self.store_to_DRAM_single_bank(0, dest_channel, bank, dest_row_index, 0, self.DRAM_column, data_tmp, op_trace)

    def broadcast_load_query(self, dic, channels_required, row_index):
        for channel in range(channels_required):
            load_tmp = []
            load_reorder = []
            if self.GQA:
                for bank in range(self.dim//self.DRAM_column):
                    load_tmp.append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index, 0, self.DRAM_column, False))
                for j in range(self.n_repeat):
                    reorder = []
                    for i in range(self.n_kv_heads):
                        reorder.append(load_tmp[i][j*128:(j+1)*128])
                    load_reorder.append(torch.cat(reorder))
            else:
                for bank in range(self.dim//self.DRAM_column):
                    load_reorder.append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index, 0, self.DRAM_column, False))
            dic[channel] = torch.cat(load_reorder)

    def _record_systolic_stage(self, kernel, stage, cycles):
        """Record fill/body/drain cycles while emitting the normal MAC opcode."""

        if cycles <= 0:
            return
        self.systolic_pipeline_cycles[stage] += cycles
        if not hasattr(self, "systolic_kernel_pipeline_cycles"):
            self.systolic_kernel_pipeline_cycles = {}
        kernel_cycles = self.systolic_kernel_pipeline_cycles.setdefault(
            kernel, {"fill": 0, "reduction": 0, "drain": 0}
        )
        kernel_cycles[stage] += cycles

    def _MAC_ABK_systolic_stage_only_trace(
        self, channels, row_index, cycles, timing, kernel, stage
    ):
        if cycles <= 0:
            return
        self._record_systolic_stage(kernel, stage, cycles)
        self.MAC_ABK_only_trace(channels, row_index, cycles, timing)

    def _trace_systolic_pipeline_only_trace(
        self,
        channels,
        row_index,
        reduction_dim,
        timing,
        kernel,
        *,
        dual_half_operand=False,
        active_rows=None,
    ):
        """Emit the migrated fill + reduction + drain command sequence."""

        if reduction_dim <= 0:
            return
        height = self.systolic_dim if active_rows is None else int(active_rows)
        if height < 1 or height > self.systolic_dim:
            raise ValueError(
                f"active systolic rows must be in [1, {self.systolic_dim}], "
                f"got {height}"
            )
        if not hasattr(self, "systolic_kernel_active_rows"):
            self.systolic_kernel_active_rows = {}
        self.systolic_kernel_active_rows.setdefault(kernel, []).append(height)
        operands = 2 if dual_half_operand else 1
        chunk_capacity = self.DRAM_column // (height * operands)
        if chunk_capacity <= 0:
            raise ValueError("systolic operands do not fit in the channel GB")
        reductions_per_dram_row = self.DRAM_column // self.burst_length
        rows_per_full_chunk = math.ceil(chunk_capacity / reductions_per_dram_row)

        for _ in range(height):
            self.WR_BIAS_only_trace(channels)
        chunks = math.ceil(reduction_dim / chunk_capacity)
        for chunk in range(chunks):
            chunk_size = min(
                chunk_capacity, reduction_dim - chunk * chunk_capacity
            )
            gb_elements = chunk_size * height * operands
            self.WR_GB_only_trace(
                channels, math.ceil(gb_elements / self.burst_length)
            )
            rows_this_chunk = math.ceil(chunk_size / reductions_per_dram_row)
            chunk_row_base = row_index + chunk * rows_per_full_chunk
            self._MAC_ABK_systolic_stage_only_trace(
                channels,
                chunk_row_base,
                height - 1,
                timing,
                kernel,
                "fill",
            )
            for row in range(rows_this_chunk):
                reduction_cycles = min(
                    reductions_per_dram_row,
                    chunk_size - row * reductions_per_dram_row,
                )
                self._MAC_ABK_systolic_stage_only_trace(
                    channels,
                    chunk_row_base + row,
                    reduction_cycles,
                    timing,
                    kernel,
                    "reduction",
                )
            self._MAC_ABK_systolic_stage_only_trace(
                channels,
                chunk_row_base + rows_this_chunk - 1,
                height - 1,
                timing,
                kernel,
                "drain",
            )
        for _ in range(height):
            self.RD_MAC_only_trace(channels)

    def Vector_Matrix_Mul_weight_systolic_tp_only_trace(
        self, projection, row_index_matrix, timing
    ):
        """Trace a standard-TP projection with batch rows streamed from GB."""

        if self.systolic_tp_layout is None:
            raise ValueError("standard systolic TP layout is not enabled")
        device_capacity = (
            self.systolic_tp_layout.total_banks * self.burst_length
        )
        rows_per_tile = math.ceil(
            projection.reduction_slice
            * self.burst_length
            / self.DRAM_column
        )
        batch_size = int(getattr(self, "batch_size", self.systolic_dim))
        for batch_start in range(0, batch_size, self.systolic_dim):
            active_rows = min(self.systolic_dim, batch_size - batch_start)
            for tile in range(projection.output_tiles):
                remaining_outputs = projection.output_dim - tile * device_capacity
                tile_outputs = min(device_capacity, remaining_outputs)
                active_channels = math.ceil(
                    tile_outputs
                    / self.systolic_tp_layout.output_columns_per_channel
                )
                if projection.reduction_groups > 1:
                    active_channels = (
                        projection.channels_per_group * projection.reduction_groups
                    )
                channels = list(range(active_channels))
                self._trace_systolic_pipeline_only_trace(
                    channels,
                    row_index_matrix + tile * rows_per_tile,
                    projection.reduction_slice,
                    timing,
                    projection.name,
                    active_rows=active_rows,
                )

    def _trace_fused_activation_systolic_pim_only_trace(self, output_dim):
        """Emit the legacy AF/RD_AF sequence for fused W1/W3.

        The standard TP layout packs W1 and W3 into one fused region, while
        the imported in-array activation applies to W1's output columns.  This
        keeps the original command behavior: one AF and one RD_AF for every
        systolic row and W1 output tile.  The command dialect has only a
        channel mask, so its partial final tile uses the full active-channel
        mask, matching the original trace implementation.
        """

        layout = self.systolic_tp_layout
        if layout is None:
            raise ValueError("standard systolic TP layout is not enabled")
        if output_dim < 1:
            return
        device_capacity = layout.total_banks * layout.burst_length
        for tile_start in range(0, output_dim, device_capacity):
            tile_outputs = min(device_capacity, output_dim - tile_start)
            active_channels = math.ceil(
                tile_outputs / layout.output_columns_per_channel
            )
            channels = list(range(active_channels))
            for _ in range(self.systolic_dim):
                self.AF_only_trace(channels)
            for _ in range(self.systolic_dim):
                self.RD_AF_only_trace(channels)

    def Vector_Matrix_Mul_score_systolic_tp_only_trace(
        self, row_index_matrix, seqlen, timing="breakdown_sa_score"
    ):
        """Local KQ trace for the standard systolic TP K-cache layout."""

        layout = self.systolic_tp_layout
        if layout is None:
            raise ValueError("standard systolic TP layout is not enabled")
        channels = list(range(layout.num_channels))
        sequence_waves = math.ceil(
            layout.k_contexts_per_bank(seqlen) / layout.burst_length
        )
        rows_per_wave = math.ceil(
            layout.shape.head_dim * layout.burst_length / layout.dram_columns
        )
        for query_wave in range(layout.qk_query_waves):
            for sequence_wave in range(sequence_waves):
                self._trace_systolic_pipeline_only_trace(
                    channels,
                    row_index_matrix + sequence_wave * rows_per_wave,
                    layout.shape.head_dim,
                    timing,
                    "qk",
                )

    def Vector_Matrix_Mul_output_systolic_tp_only_trace(
        self, row_index_matrix, seqlen, timing="breakdown_sa_output"
    ):
        """Local SV trace using both independent 8-lane burst halves."""

        layout = self.systolic_tp_layout
        if layout is None:
            raise ValueError("standard systolic TP layout is not enabled")
        sv = layout.sv(seqlen)
        channels = list(range(layout.num_channels))
        for _query_wave in range(sv.query_waves):
            self._trace_systolic_pipeline_only_trace(
                channels,
                row_index_matrix,
                sv.contexts_per_half,
                timing,
                "sv_" + sv.mode,
                dual_half_operand=True,
            )

    def Vector_Matrix_Mul_flash_attention_systolic_tp_only_trace(
        self, k_cache_row_index_matrix, v_cache_row_index_matrix, seqlen
    ):
        """Trace cent_dev-style block FlashAttention for the TP layout.

        Each context block drains its QK result and immediately consumes the
        normalized score stream in SV.  As in cent_dev, the online-softmax
        state and the RD_MAC-to-WR_GB hand-off are represented by the
        analytical accelerator model rather than explicit PNM commands.  In
        particular, no score workspace is materialized with W_MEM/R_MEM.

        The block row strides deliberately follow the imported cent_dev
        schedule.  They are command-address strides for the trace dialect,
        not a separate replacement for the Device--Channel--Bank cache
        placement used for KV cache writes.
        """

        layout = self.systolic_tp_layout
        if layout is None:
            raise ValueError("standard systolic TP layout is not enabled")
        if seqlen < 1:
            return

        block_size = self.flash_attention_block_size
        flash_attention_blocks = math.ceil(seqlen / block_size)
        k_rows_per_vector = math.ceil(
            layout.shape.head_dim * layout.shape.local_kv_heads
            / layout.dram_columns
        )
        k_sequence_iterations = math.ceil(block_size / layout.total_banks)
        k_rows_per_block = k_sequence_iterations * k_rows_per_vector
        v_rows_per_block = math.ceil(block_size / layout.dram_columns)

        for flash_attention_block in range(flash_attention_blocks):
            block_seqlen = min(
                block_size,
                seqlen - flash_attention_block * block_size,
            )
            k_cache_row_index = (
                k_cache_row_index_matrix
                + flash_attention_block * k_rows_per_block
            )
            v_cache_row_index = (
                v_cache_row_index_matrix
                + flash_attention_block * v_rows_per_block
            )
            self.Vector_Matrix_Mul_score_systolic_tp_only_trace(
                k_cache_row_index, block_seqlen
            )
            if not self.trace_score:
                self.Vector_Matrix_Mul_output_systolic_tp_only_trace(
                    v_cache_row_index, block_seqlen
                )

    def systolic_tp_cache_rows_per_batch(self):
        """Return the contiguous per-request K/V row strides for this TP rank."""

        layout = self.systolic_tp_layout
        if layout is None:
            raise ValueError("standard systolic TP layout is not enabled")
        sv = layout.sv(self.max_seq_len)
        k_rows = self.kv_head_layout.k_rows_per_bank
        v_rows = math.ceil(
            sv.contexts_per_half * self.burst_length / self.DRAM_column
        )
        return k_rows, v_rows

    def trace_systolic_tp_cache_update(
        self, row_index_k, row_index_v, sequence, batch_index=0
    ):
        """Trace one token write into one contiguous batch K/V cache block."""

        layout = self.systolic_tp_layout
        if layout is None:
            raise ValueError("standard systolic TP layout is not enabled")
        if batch_index < 0 or batch_index >= self.batch_size:
            raise ValueError(
                f"batch index {batch_index} is outside batch size {self.batch_size}"
            )
        k_rows_per_batch, v_rows_per_batch = self.systolic_tp_cache_rows_per_batch()
        row_index_k += batch_index * k_rows_per_batch
        row_index_v += batch_index * v_rows_per_batch
        # K keeps the existing standard-TP sequence-to-bank mapping.
        for local_head in range(layout.shape.local_kv_heads):
            channel, bank, row, _segment = self.kv_head_layout.k_location(
                local_head, sequence
            )
            self.W_MEM_only_trace(channel, bank, row_index_k + row, self.head_dim)

        sv = layout.sv(self.max_seq_len)
        bundles_per_row = layout.dram_columns // layout.burst_length
        if sv.mode == "paired_kv_heads":
            pairs = layout.shape.local_kv_heads // 2
            context_group = sequence % sv.context_groups_per_kv_head
            context_in_group = sequence // sv.context_groups_per_kv_head
            row = context_in_group // bundles_per_row
            for pair in range(pairs):
                first_channel = (
                    pair * sv.channels_per_work_item
                    + context_group * sv.channels_per_context_group
                )
                for channel in range(
                    first_channel,
                    first_channel + sv.channels_per_context_group,
                ):
                    for bank in range(layout.banks_per_channel):
                        self.W_MEM_only_trace(
                            channel, bank, row_index_v + row, layout.burst_length
                        )
        elif sv.mode == "paired_query_groups":
            context_group = sequence % sv.context_groups_per_kv_head
            context_in_group = sequence // sv.context_groups_per_kv_head
            row = context_in_group // bundles_per_row
            first_channel = context_group * sv.channels_per_context_group
            for channel in range(
                first_channel,
                first_channel + sv.channels_per_context_group,
            ):
                for bank in range(layout.banks_per_channel):
                    self.W_MEM_only_trace(
                        channel, bank, row_index_v + row, layout.burst_length
                    )
        else:
            context_group = sequence % sv.context_groups_per_kv_head
            physical_groups = (
                layout.num_channels // sv.channels_per_context_group
            )
            physical_group = context_group % physical_groups
            context_bundle = sequence // sv.context_groups_per_kv_head
            row = context_bundle // bundles_per_row
            first_channel = physical_group * sv.channels_per_context_group
            for channel in range(
                first_channel,
                first_channel + sv.channels_per_context_group,
            ):
                for bank in range(layout.banks_per_channel):
                    self.W_MEM_only_trace(
                        channel, bank, row_index_v + row, layout.half_width
                    )

    def Vector_Matrix_Mul_weight_systolic_pim_only_trace(self, channel_lst, row_index_matrix, vector_dim, matrix_col, activation_function=False):
        matrix_cols = matrix_col     # 1024: Wk in Llama3 70B
        matrix_rows = vector_dim     # 8192: Wk in Llama3 70B
        matrix_cols_per_bank = math.ceil(matrix_cols / self.FC_total_banks)
        banks_per_matrix_col = math.ceil(self.burst_length / matrix_cols_per_bank)
        # print("matrix_cols", matrix_cols, "matrix_rows", matrix_rows, "total banks", self.FC_total_banks, "matrix_cols_per_bank", matrix_cols_per_bank, "banks_per_matrix_col", banks_per_matrix_col)
        matrix_cols_per_bank = max(matrix_cols_per_bank, self.burst_length)
        bursts_of_matrix_cols_per_bank = math.ceil(matrix_cols_per_bank / self.burst_length)
        matrix_cols_per_bank_padding = bursts_of_matrix_cols_per_bank * self.burst_length  # round up to multiple of burst_length
        bursts_per_DRAM_row = self.DRAM_column // self.burst_length
        # assert matrix_rows // banks_per_matrix_col * self.burst_length >= self.DRAM_column, "Each bank has more than one row of data"
        systolic_dim_padding = self.systolic_dim
        chunk_size = self.DRAM_column // systolic_dim_padding # 1K / 8 = 128
        chunks = math.ceil(vector_dim / chunk_size)     # 4K / 128 = 32
        DRAM_rows_per_chunk = math.ceil(chunk_size / bursts_per_DRAM_row)

        if banks_per_matrix_col > 1:    # Partition matrix_rows into banks
            banks_per_matrix_row = matrix_cols // self.burst_length
            banks_per_matrix_col = self.FC_total_banks // banks_per_matrix_row
            utilized_banks = banks_per_matrix_row * banks_per_matrix_col
            matrix_rows_per_bank = math.ceil(matrix_rows / banks_per_matrix_col)
            bursts_per_bank = math.ceil(matrix_rows_per_bank * matrix_cols_per_bank_padding / self.burst_length)
            DRAM_rows_per_bank = math.ceil(bursts_per_bank / bursts_per_DRAM_row)
            channels_utilized = self.channels_per_block
            channels_required_all_devices = (utilized_banks - 1) // self.num_banks + 1
            if channels_required_all_devices > self.num_channels:
                channel_multi_transformer_block_required = self.num_channels
            else:
                channel_multi_transformer_block_required = channels_required_all_devices if self.single_tb_per_device else self.num_channels // channels_required_all_devices * channels_required_all_devices
            channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]

            channel = 0
            matrix_row_block_index = channel * self.num_banks // banks_per_matrix_row
            if matrix_row_block_index == banks_per_matrix_col - 1:
                matrix_rows_per_bank_per_row_block = matrix_rows - matrix_row_block_index * matrix_rows_per_bank
            else:
                matrix_rows_per_bank_per_row_block = matrix_rows_per_bank
            chunks_per_row_block = math.ceil(matrix_rows_per_bank_per_row_block / chunk_size)

            for _ in range(systolic_dim_padding):
                self.WR_BIAS_only_trace(channel_lst)
            for chunk_index in range(chunks_per_row_block):
                if chunk_index == chunks_per_row_block - 1:
                    vector_size = matrix_rows_per_bank_per_row_block - chunk_index * chunk_size
                else:
                    vector_size = chunk_size
                DRAM_rows_this_chunk = math.ceil(vector_size / bursts_per_DRAM_row) # the last chunk may only have one DRAM row
                GB_op_size = math.ceil(vector_size * systolic_dim_padding / self.burst_length)
                self.WR_GB_only_trace(channel_lst, GB_op_size)

                # Systolic begin padding
                DRAM_row = 0
                row_offset = chunk_index * DRAM_rows_per_chunk + DRAM_row
                self.MAC_ABK_only_trace(channel_lst, row_index_matrix + row_offset, systolic_dim_padding - 1)

                for DRAM_row in range(DRAM_rows_this_chunk):  # not considering fragmentation
                    if DRAM_row == DRAM_rows_this_chunk - 1:
                        op_size = vector_size - DRAM_row * bursts_per_DRAM_row
                    else:
                        op_size = bursts_per_DRAM_row
                    row_offset = chunk_index * DRAM_rows_per_chunk + DRAM_row
                    self.MAC_ABK_only_trace(channel_lst, row_index_matrix + row_offset, op_size)

                # Systolic end padding
                DRAM_row = DRAM_rows_this_chunk - 1
                row_offset = chunk_index * DRAM_rows_per_chunk + DRAM_row
                self.MAC_ABK_only_trace(channel_lst, row_index_matrix + row_offset, systolic_dim_padding - 1)

            if activation_function:
                for _ in range(systolic_dim_padding):
                    self.AF_only_trace(channel_lst)
                for _ in range(systolic_dim_padding):
                    self.RD_AF_only_trace(channel_lst)
            else:
                for _ in range(systolic_dim_padding):
                    self.RD_MAC_only_trace(channel_lst)

        else:   # Keep all matrix_rows in a single bank, >=1 burst per bank
            utilized_banks = math.ceil(matrix_cols / matrix_cols_per_bank_padding)
            matrix_rows_per_bank = math.ceil(matrix_rows / banks_per_matrix_col)
            banks_per_matrix_row = utilized_banks
            bursts_per_bank = math.ceil(matrix_rows_per_bank * matrix_cols_per_bank_padding / self.burst_length)
            DRAM_rows_per_bank = math.ceil(bursts_per_bank / bursts_per_DRAM_row)
            DRAM_rows_per_burst_block = math.ceil(matrix_rows_per_bank / bursts_per_DRAM_row)
            channels_utilized = self.channels_per_block
            channels_required_all_devices = (utilized_banks - 1) // self.num_banks + 1
            if channels_required_all_devices > self.num_channels:
                channel_multi_transformer_block_required = self.num_channels
            else:
                channel_multi_transformer_block_required = channels_required_all_devices if self.single_tb_per_device else self.num_channels // channels_required_all_devices * channels_required_all_devices
            channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]

            for burst in range(bursts_of_matrix_cols_per_bank):
                for _ in range(systolic_dim_padding):
                    self.WR_BIAS_only_trace(channel_lst)
                for chunk_index in range(chunks):
                    if chunk_index == chunks - 1:
                        vector_size = vector_dim - chunk_index * chunk_size
                    else:
                        vector_size = chunk_size
                    GB_op_size = math.ceil(vector_size * systolic_dim_padding / self.burst_length)
                    self.WR_GB_only_trace(channel_lst, GB_op_size)

                    # Systolic begin padding
                    DRAM_row = 0
                    row_offset = burst * DRAM_rows_per_burst_block + chunk_index * DRAM_rows_per_chunk + DRAM_row
                    self.MAC_ABK_only_trace(channel_lst, row_index_matrix + row_offset, systolic_dim_padding - 1)

                    for DRAM_row in range(DRAM_rows_per_chunk):  # not considering fragmentation
                        row_offset = burst * DRAM_rows_per_burst_block + chunk_index * DRAM_rows_per_chunk + DRAM_row
                        self.MAC_ABK_only_trace(channel_lst, row_index_matrix + row_offset, bursts_per_DRAM_row)

                    # Systolic end padding
                    DRAM_row = DRAM_rows_per_chunk - 1
                    row_offset = burst * DRAM_rows_per_burst_block + chunk_index * DRAM_rows_per_chunk + DRAM_row
                    self.MAC_ABK_only_trace(channel_lst, row_index_matrix + row_offset, systolic_dim_padding - 1)

                if activation_function:
                    for _ in range(systolic_dim_padding):
                        self.AF_only_trace(channel_lst)
                    for _ in range(systolic_dim_padding):
                        self.RD_AF_only_trace(channel_lst)
                else:
                    for _ in range(systolic_dim_padding):
                        self.RD_MAC_only_trace(channel_lst)

    def Vector_Matrix_Mul_flash_attention_pim_only_trace(self, K_cache_row_index_matrix, V_cache_row_index_matrix, seqlen):

        if self.model_parallel and not self.intra_device_attention:
            banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1                   # 8,  32, 256, 2k
            rows_per_seq_iteration = (banks_per_head - 1) // self.head_dim + 1                  # 1,  1,  2,   16
            seq_iterations = (seqlen - 1) // (self.DRAM_column * rows_per_seq_iteration) + 1    # 32, 32, 16,  2
            seqlen_per_seq_iteration = (seqlen - 1) // seq_iterations + 1
            self.flash_attention_block_size = max(self.flash_attention_block_size, seqlen_per_seq_iteration)

        flash_attention_blocks = (seqlen - 1) // self.flash_attention_block_size + 1
        for flash_attention_block in range(flash_attention_blocks):
            seqlen_flash_attention_block = seqlen - flash_attention_block * self.flash_attention_block_size if flash_attention_block == flash_attention_blocks - 1 else self.flash_attention_block_size

            # Q x K cache
            K_cache_rows_per_vector = (self.head_dim * self.n_kv_heads - 1) // self.DRAM_column + 1
            K_cache_seq_iterations = (self.flash_attention_block_size - 1) // self.FC_total_banks + 1
            K_cache_row_index = K_cache_row_index_matrix + K_cache_seq_iterations * K_cache_rows_per_vector * flash_attention_block

            # Score x V cache
            rows_per_flash_attention_block = (self.flash_attention_block_size - 1) // (self.DRAM_column) + 1
            V_cache_row_index = V_cache_row_index_matrix + rows_per_flash_attention_block * flash_attention_block

            if self.systolic_pim:
                if self.MLA:
                    self.Vector_Matrix_Mul_score_systolic_pim_only_trace(K_cache_row_index, seqlen_flash_attention_block, 64, 1, self.n_heads)
                    self.Vector_Matrix_Mul_score_systolic_pim_only_trace(K_cache_row_index, seqlen_flash_attention_block, 512, 1, 1)
                    # use head_dim=128 and 4 heads because when sa=8, 8x128 can fit in the global buffer, 8x512 cannot fit
                else:
                    self.Vector_Matrix_Mul_score_systolic_pim_only_trace(K_cache_row_index, seqlen_flash_attention_block)
                if not self.trace_score:
                    self.Vector_Matrix_Mul_output_systolic_pim_only_trace(V_cache_row_index, seqlen_flash_attention_block)
            else:
                if self.MLA:
                    self.Vector_Matrix_Mul_score_pim_only_trace(K_cache_row_index, seqlen_flash_attention_block, 64, 1, self.n_heads)
                    self.Vector_Matrix_Mul_score_pim_only_trace(K_cache_row_index, seqlen_flash_attention_block, 512, 1, 1)
                else:
                    self.Vector_Matrix_Mul_score_pim_only_trace(K_cache_row_index, seqlen_flash_attention_block)
                if not self.trace_score:
                    self.Vector_Matrix_Mul_output_pim_only_trace(V_cache_row_index, seqlen_flash_attention_block)

    def Vector_Matrix_Mul_score_systolic_pim_only_trace(self, row_index_matrix, seqlen, deepseek_dim=64, deepseek_kv_heads=1, deepseek_n_head=1):
        head_dim = deepseek_dim if self.MLA else self.head_dim
        n_kv_heads = deepseek_kv_heads if self.MLA else self.n_kv_heads
        n_heads = deepseek_n_head if self.MLA else self.n_heads
        GQA = n_heads // n_kv_heads
        DRAM_rows_per_head = head_dim * self.burst_length // self.DRAM_column   # 2
        total_bursts = math.ceil(seqlen / self.burst_length)
        seq_iterations = math.ceil(total_bursts / self.FC_total_banks)
        bursts_per_DRAM_row = self.DRAM_column // self.burst_length
        systolic_dim_padding = min(self.systolic_dim, GQA)
        GQA_iterations = (GQA - 1) // systolic_dim_padding + 1
        channels_utilized = self.channels_per_block
        channels_required_all_devices = self.FC_total_banks // self.num_banks
        if channels_required_all_devices > self.num_channels:
            channel_multi_transformer_block_required = self.num_channels
        else:
            channel_multi_transformer_block_required = channels_required_all_devices if self.single_tb_per_device else self.num_channels // channels_required_all_devices * channels_required_all_devices
        channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]

        for i in range(self.batch_size):
            for _ in range(GQA_iterations):
                for kv_head in range(n_kv_heads):
                    for seq_iter in range(seq_iterations):
                        GB_op_size = GQA * head_dim // self.burst_length
                        self.WR_GB_only_trace(channel_lst, GB_op_size)
                        for _ in range(systolic_dim_padding):
                            self.WR_BIAS_only_trace(channel_lst)

                        # Systolic begin padding
                        DRAM_row = 0
                        row_offset = i * seq_iterations * n_kv_heads * DRAM_rows_per_head + seq_iter * n_kv_heads * DRAM_rows_per_head + kv_head * DRAM_rows_per_head + DRAM_row
                        self.MAC_ABK_only_trace(channel_lst, row_index_matrix + row_offset, systolic_dim_padding - 1)

                        for DRAM_row in range(DRAM_rows_per_head):
                            row_offset = i * seq_iterations * n_kv_heads * DRAM_rows_per_head + seq_iter * n_kv_heads * DRAM_rows_per_head + kv_head * DRAM_rows_per_head + DRAM_row
                            self.MAC_ABK_only_trace(channel_lst, row_index_matrix + row_offset, bursts_per_DRAM_row)

                        # Systolic end padding
                        DRAM_row = DRAM_rows_per_head - 1
                        row_offset = i * seq_iterations * n_kv_heads * DRAM_rows_per_head + seq_iter * n_kv_heads * DRAM_rows_per_head + kv_head * DRAM_rows_per_head + DRAM_row
                        self.MAC_ABK_only_trace(channel_lst, row_index_matrix + row_offset, systolic_dim_padding - 1)

                        for _ in range(systolic_dim_padding):
                            self.RD_MAC_only_trace(channel_lst)

    def Vector_Matrix_Mul_output_systolic_pim_only_trace(self, row_index_matrix, seqlen):
        # first scale down Deepseek's kv_lora_rank from 512 to 128, aligning with other models, then scale up in pairs_pf_kv_heads
        head_dim = self.head_dim if not self.MLA else self.kv_lora_rank // 4
        n_kv_heads = self.n_kv_heads
        GQA = self.n_repeat
        num_banks_per_head_dim = head_dim // self.burst_length         # 8
        heads_per_channel = self.num_banks // num_banks_per_head_dim        # 2
        num_bursts_per_DRAM_row = self.DRAM_column // self.burst_length     # 64
        # Variable
        systolic_dim_padding = min(self.systolic_dim, GQA)
        GQA_iterations = (GQA - 1) // self.systolic_dim + 1
        # systolic_pim = 8, chunk_size = 1024 // 8 = 128
        # systolic_pim = 1, chunk_size = 1024 // 8 = 1024
        chunk_size = self.DRAM_column // systolic_dim_padding
        chunks = (seqlen - 1) // chunk_size + 1
        # each channel has 16 banks, executing one KV head, if there num_channels < n_kv_heads, each channel has multiple KV heads, otherwise each channel has one KV head and chunks can be scaled down based on the number of channels
        if self.MLA:
            pairs_of_kv_heads = n_kv_heads * 4 // heads_per_channel     # 2 in DeepSeek
        else:
            pairs_of_kv_heads = n_kv_heads // heads_per_channel    # 4 in Llama-70B and 16 in Llama-7B
        if self.channels_per_block < pairs_of_kv_heads:
            paired_kv_head_per_channel = math.ceil(pairs_of_kv_heads / self.channels_per_block)
            seqlen_iterations = 1
        else:
            paired_kv_head_per_channel = 1
            seqlen_iterations = self.channels_per_block // pairs_of_kv_heads
        channels_per_seqlen_iteration = math.ceil(pairs_of_kv_heads / paired_kv_head_per_channel)
        channels_utilized = channels_per_seqlen_iteration * seqlen_iterations
        chunks_per_seqlen_iteration = math.ceil(chunks / seqlen_iterations)
        results = torch.zeros(torch.Size([self.batch_size, self.n_heads, head_dim]))
        channels_required_all_devices = seqlen_iterations * channels_per_seqlen_iteration
        if channels_required_all_devices > self.num_channels:
            channel_multi_transformer_block_required = self.num_channels
        else:
            channel_multi_transformer_block_required = channels_required_all_devices if self.single_tb_per_device else self.num_channels // channels_required_all_devices * channels_required_all_devices
        channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]

        for batch_index in range(self.batch_size):
            for _ in range(GQA_iterations):
                channel_index = 0
                for paired_kv_head in range(paired_kv_head_per_channel):
                    paired_kv_head_index = (channel_index * paired_kv_head_per_channel) % pairs_of_kv_heads + paired_kv_head
                    seqlen_iteration = 0
                    seqlen_per_iteration = chunks_per_seqlen_iteration * chunk_size
                    for _ in range(systolic_dim_padding):
                        self.WR_BIAS_only_trace(channel_lst)
                    for chunk in range(chunks_per_seqlen_iteration):
                        chunk_index = seqlen_iteration * chunks_per_seqlen_iteration + chunk
                        if chunk < chunks_per_seqlen_iteration - 1:
                            seqlen_per_chunk = chunk_size
                        else:
                            if chunk_index >= chunks:
                                break
                            seqlen_per_chunk = seqlen_per_iteration - (chunks_per_seqlen_iteration - 1) * chunk_size
                        rows_per_chunk = math.ceil(seqlen_per_chunk * self.burst_length / self.DRAM_column)

                        # Systolic begin padding
                        row = 0
                        row_offset = batch_index * paired_kv_head_per_channel * seqlen_iterations * chunks_per_seqlen_iteration * rows_per_chunk + paired_kv_head * seqlen_iterations * chunks_per_seqlen_iteration * rows_per_chunk + seqlen_iteration * chunks_per_seqlen_iteration * rows_per_chunk + chunk * rows_per_chunk + row
                        self.MAC_ABK_only_trace(channel_lst, row_index_matrix + row_offset, systolic_dim_padding - 1)

                        for row in range(rows_per_chunk):
                            if row == rows_per_chunk - 1:
                                row_len = seqlen_per_chunk * self.burst_length - row * self.DRAM_column
                            else:
                                row_len = self.DRAM_column
                            left_seqlen = row_len // self.burst_length
                            vector_WR_GB_size = left_seqlen
                            GB_op_size = math.ceil(vector_WR_GB_size / self.burst_length)
                            self.WR_GB_only_trace(channel_lst, GB_op_size)
                            row_offset = batch_index * paired_kv_head_per_channel * seqlen_iterations * chunks_per_seqlen_iteration * rows_per_chunk + paired_kv_head * seqlen_iterations * chunks_per_seqlen_iteration * rows_per_chunk + seqlen_iteration * chunks_per_seqlen_iteration * rows_per_chunk + chunk * rows_per_chunk + row
                            self.MAC_ABK_only_trace(channel_lst, row_index_matrix + row_offset, left_seqlen)

                        # Systolic end padding
                        row = rows_per_chunk - 1
                        row_offset = batch_index * paired_kv_head_per_channel * seqlen_iterations * chunks_per_seqlen_iteration * rows_per_chunk + paired_kv_head * seqlen_iterations * chunks_per_seqlen_iteration * rows_per_chunk + seqlen_iteration * chunks_per_seqlen_iteration * rows_per_chunk + chunk * rows_per_chunk + row
                        self.MAC_ABK_only_trace(channel_lst, row_index_matrix + row_offset, systolic_dim_padding - 1)

                    for _ in range(systolic_dim_padding):
                        self.RD_MAC_only_trace(channel_lst)

    def Vector_Matrix_Mul_weight_pim_only_trace(self, channel_lst, row_index_matrix, vector_dim, matrix_col, total_banks, timing):
        matrix_col_per_bank = (matrix_col - 1) // total_banks + 1
        rows_per_vector = (vector_dim - 1) // self.DRAM_column + 1
        utilized_banks = (matrix_col - 1) // matrix_col_per_bank + 1  # shape = [4096, 11008]
        channels_required_all_devices = (utilized_banks - 1) // self.num_banks + 1
        channel_multi_transformer_block_required = 32 if channels_required_all_devices > 32 else self.num_channels // channels_required_all_devices * channels_required_all_devices
        channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
        if self.GEMV_order == "no-reuse":
            for row_index in range(rows_per_vector):
                if row_index == rows_per_vector - 1:
                    op_size = (vector_dim - self.DRAM_column * row_index - 1) // self.burst_length + 1
                else:
                    op_size = self.DRAM_column // self.burst_length
                self.WR_GB_only_trace(channel_lst, op_size)
                for vector_index_per_bank in range(matrix_col_per_bank):
                    self.WR_BIAS_only_trace(channel_lst)
                    self.MAC_ABK_only_trace(channel_lst, row_index_matrix + vector_index_per_bank * rows_per_vector + row_index, op_size, timing)
                    self.RD_MAC_only_trace(channel_lst)
        elif self.GEMV_order == "reuse-GB":
            num_reuse_groups = (matrix_col_per_bank - 1) // self.reuse_size + 1
            reuse_group_size = (matrix_col_per_bank - 1) // num_reuse_groups + 1
            for row_index in range(rows_per_vector):
                if row_index == rows_per_vector - 1:
                    op_size = (vector_dim - self.DRAM_column * row_index - 1) // self.burst_length + 1
                else:
                    op_size = self.DRAM_column // self.burst_length
                self.WR_GB_only_trace(channel_lst, op_size)
                for reuse_group_index in range(num_reuse_groups):
                    if reuse_group_index < num_reuse_groups - 1:
                        num_left_maxtrix_col = reuse_group_size
                    else:
                        num_left_maxtrix_col = matrix_col_per_bank - reuse_group_size * (num_reuse_groups - 1)
                    for latch_index in range(num_left_maxtrix_col):
                        self.WR_BIAS_only_trace(channel_lst)
                    for latch_index in range(num_left_maxtrix_col):
                        vector_index_per_bank = reuse_group_index * reuse_group_size + latch_index
                        self.MAC_ABK_only_trace(channel_lst, row_index_matrix + vector_index_per_bank * rows_per_vector + row_index, op_size, timing)
                    for latch_index in range(num_left_maxtrix_col):
                        self.RD_MAC_only_trace(channel_lst)

    def Vector_Matrix_Mul_weight_af_pim_only_trace(self, channel_lst, row_index_matrix, vector_dim, matrix_col, total_banks, timing):
        matrix_col_per_bank = (matrix_col - 1) // total_banks + 1
        rows_per_vector = (vector_dim - 1) // self.DRAM_column + 1
        utilized_banks = (matrix_col - 1) // matrix_col_per_bank + 1  # shape = [4096, 11008]
        channels_required_all_devices = (utilized_banks - 1) // self.num_banks + 1
        channel_multi_transformer_block_required = 32 if channels_required_all_devices > 32 else self.num_channels // channels_required_all_devices * channels_required_all_devices
        channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
        if self.GEMV_order == "no-reuse":
            for row_index in range(rows_per_vector):
                if row_index == rows_per_vector - 1:
                    op_size = (vector_dim - self.DRAM_column * row_index - 1) // self.burst_length + 1
                else:
                    op_size = self.DRAM_column // self.burst_length
                self.WR_GB_only_trace(channel_lst, op_size)
                for vector_index_per_bank in range(matrix_col_per_bank):
                    self.WR_BIAS_only_trace(channel_lst)
                    self.MAC_ABK_only_trace(channel_lst, row_index_matrix + vector_index_per_bank * rows_per_vector + row_index, op_size, timing)
                    if row_index == rows_per_vector - 1:
                        self.AF_only_trace(channel_lst)
                    self.RD_MAC_only_trace(channel_lst)
        elif self.GEMV_order == "reuse-GB":
            num_reuse_groups = (matrix_col_per_bank - 1) // (self.reuse_size // 2) + 1
            reuse_group_size = (matrix_col_per_bank - 1) // num_reuse_groups + 1
            for row_index in range(rows_per_vector):
                if row_index == rows_per_vector - 1:
                    op_size = (vector_dim - self.DRAM_column * row_index - 1) // self.burst_length + 1
                else:
                    op_size = self.DRAM_column // self.burst_length
                self.WR_GB_only_trace(channel_lst, op_size)
                for reuse_group_index in range(num_reuse_groups):
                    if reuse_group_index < num_reuse_groups - 1:
                        num_left_maxtrix_col = reuse_group_size
                    else:
                        num_left_maxtrix_col = matrix_col_per_bank - reuse_group_size * (num_reuse_groups - 1)
                    for latch_index in range(num_left_maxtrix_col):
                        self.WR_BIAS_only_trace(channel_lst)
                    for latch_index in range(num_left_maxtrix_col):
                        vector_index_per_bank = reuse_group_index * reuse_group_size + latch_index
                        self.MAC_ABK_only_trace(channel_lst, row_index_matrix + vector_index_per_bank * rows_per_vector + row_index, op_size, timing)
                    if row_index == rows_per_vector - 1:
                        for latch_index in range(num_left_maxtrix_col):
                            self.AF_only_trace(channel_lst)
                            self.RD_AF_only_trace(channel_lst)
                    for latch_index in range(num_left_maxtrix_col):
                        self.RD_MAC_only_trace(channel_lst)
            
    def Vector_Matrix_Mul_kv_projection_pim_only_trace(
        self,
        row_index_k,
        row_index_v,
        vector_dim,
        timing,
    ):
        """Trace K and V projections on disjoint channel halves.

        The two command streams are interleaved so Ramulator can overlap work
        on independent channels.  Every selected channel uses all of its banks;
        TP=8 70B therefore uses eight full channels for K and eight for V rather
        than assuming an unavailable per-bank MAC mask.
        """

        if not self.kv_head_tp:
            raise ValueError("the split K/V projection is specific to KV-head TP")
        layout = self.kv_head_layout
        active_channels = layout.kv_projection_channels_per_operand
        half = self.num_channels // 2
        k_channels = list(range(active_channels))
        v_channels = list(range(half, half + active_channels))
        outputs_per_bank = layout.kv_projection_columns_per_bank
        rows_per_vector = math.ceil(vector_dim / self.DRAM_column)
        num_reuse_groups = math.ceil(outputs_per_bank / self.reuse_size)
        reuse_group_size = math.ceil(outputs_per_bank / num_reuse_groups)

        for vector_row in range(rows_per_vector):
            remaining = vector_dim - vector_row * self.DRAM_column
            op_size = math.ceil(min(self.DRAM_column, remaining) / self.burst_length)
            self.WR_GB_only_trace(k_channels, op_size)
            self.WR_GB_only_trace(v_channels, op_size)
            for group in range(num_reuse_groups):
                group_outputs = min(
                    reuse_group_size,
                    outputs_per_bank - group * reuse_group_size,
                )
                for _ in range(group_outputs):
                    self.WR_BIAS_only_trace(k_channels)
                    self.WR_BIAS_only_trace(v_channels)
                for output_in_group in range(group_outputs):
                    output_index = group * reuse_group_size + output_in_group
                    row_offset = output_index * rows_per_vector + vector_row
                    self.MAC_ABK_only_trace(
                        k_channels, row_index_k + row_offset, op_size, timing
                    )
                    self.MAC_ABK_only_trace(
                        v_channels, row_index_v + row_offset, op_size, timing
                    )
                for _ in range(group_outputs):
                    self.RD_MAC_only_trace(k_channels)
                    self.RD_MAC_only_trace(v_channels)

    def trace_kv_head_cache_update(self, row_index_k, row_index_v, sequence):
        """Trace the decode token's K/V write into the new physical layouts."""

        if not self.kv_head_tp:
            raise ValueError("KV-head cache update requires KV-head TP")
        layout = self.kv_head_layout
        for local_head in range(layout.shape.local_kv_heads):
            channel, bank, row, _segment = layout.k_location(local_head, sequence)
            self.W_MEM_only_trace(
                channel,
                bank,
                row_index_k + row,
                self.head_dim,
            )

            writes = {}
            for dimension in range(self.head_dim):
                v_channel, v_bank, v_row, _column = layout.v_location(
                    local_head, dimension, sequence
                )
                key = (v_channel, v_bank, v_row)
                writes[key] = writes.get(key, 0) + 1
            for (v_channel, v_bank, v_row), elements in writes.items():
                self.W_MEM_only_trace(
                    v_channel,
                    v_bank,
                    row_index_v + v_row,
                    elements,
                )

    def _Vector_Matrix_Mul_score_kv_head_tp_only_trace(
        self, row_index_matrix, seqlen, timing
    ):
        """QK with eight independently accumulated K vectors per DRAM row."""

        layout = self.kv_head_layout
        banks_per_head = layout.banks_per_kv_head
        wave_span = banks_per_head * layout.k_contexts_per_row
        waves = math.ceil(seqlen / wave_span)
        q_op_size = self.head_dim // self.burst_length

        head_channels = [
            list(layout.kv_head_channels(local_head))
            for local_head in range(layout.shape.local_kv_heads)
        ]
        for _query_in_group in range(layout.shape.gqa_factor):
            # Each local KV head owns a disjoint channel mask.  Populate all
            # channel-local GBs before issuing the same query slot across the
            # heads, then use RD_MAC as the cross-mask completion barrier.
            for channels in head_channels:
                self.WR_GB_only_trace(channels, q_op_size)
            for wave in range(waves):
                wave_base = wave * wave_span
                for segment in range(layout.k_contexts_per_row):
                    first_sequence = wave_base + segment * banks_per_head
                    if first_sequence >= seqlen:
                        break
                    valid_banks = min(banks_per_head, seqlen - first_sequence)
                    valid_channels = math.ceil(valid_banks / self.num_banks)
                    active_masks = [
                        channels[:valid_channels] for channels in head_channels
                    ]
                    active_channels = [
                        channel
                        for channels in active_masks
                        for channel in channels
                    ]
                    # The channel-local GBs contain different queries, but the
                    # MAC opcode, row, and length are identical.  One union
                    # mask therefore launches every local head concurrently.
                    self.WR_BIAS_only_trace(active_channels)
                    self.MAC_ABK_only_trace(
                        active_channels,
                        row_index_matrix + wave,
                        q_op_size,
                        timing,
                    )
                    self.RD_MAC_only_trace(active_channels)

    def _Vector_Matrix_Mul_output_kv_head_tp_only_trace(
        self, row_index_matrix, seqlen, timing
    ):
        """SV with context shards placed in independent channel groups.

        A split V head uses one 128-bank channel group per context shard.  The
        group shared GB therefore holds exactly that shard's score slice and
        MAC_ABK computes all 128 output dimensions in parallel.  Cross-group
        vector addition is intentionally accounted for in postprocessing.
        """

        layout = self.kv_head_layout
        active_rows_per_dimension = math.ceil(seqlen / self.DRAM_column)
        physical_rows_per_dimension = math.ceil(
            self.max_seq_len / self.DRAM_column
        )

        head_channels = [
            list(layout.kv_head_channels(local_head))
            for local_head in range(layout.shape.local_kv_heads)
        ]
        for _query_in_group in range(layout.shape.gqa_factor):
            if not layout.v_context_split:
                for dim_iteration in range(layout.v_dimension_iterations):
                    active_channels = [
                        channel
                        for channels in head_channels
                        for channel in channels
                    ]
                    self.WR_BIAS_only_trace(active_channels)
                    for row in range(active_rows_per_dimension):
                        elements = min(
                            self.DRAM_column,
                            seqlen - row * self.DRAM_column,
                        )
                        op_size = math.ceil(elements / self.burst_length)
                        for channels in head_channels:
                            self.WR_GB_only_trace(channels, op_size)
                        self.MAC_ABK_only_trace(
                            active_channels,
                            row_index_matrix
                            + dim_iteration * physical_rows_per_dimension
                            + row,
                            op_size,
                            timing,
                        )
                    self.RD_MAC_only_trace(active_channels)
                continue

            # A work item is one (KV head, context shard) pair.  The masks of
            # all such pairs are disjoint, so all heads and all active context
            # shards can keep their channel groups in flight together.
            shard_capacity = layout.v_context_capacity_per_bank
            work_items = []
            for local_head in range(layout.shape.local_kv_heads):
                for context_shard in range(layout.v_context_shards):
                    shard_start = context_shard * shard_capacity
                    if shard_start >= seqlen:
                        break
                    shard_elements = min(
                        shard_capacity, seqlen - shard_start
                    )
                    work_items.append(
                        (
                            list(
                                layout.v_shard_channels(
                                    local_head, context_shard
                                )
                            ),
                            shard_elements,
                        )
                    )

            active_channels = [
                channel
                for channels, _shard_elements in work_items
                for channel in channels
            ]
            self.WR_BIAS_only_trace(active_channels)
            max_active_rows = max(
                math.ceil(shard_elements / self.DRAM_column)
                for _channels, shard_elements in work_items
            )
            for row in range(max_active_rows):
                mac_channels_by_op_size = {}
                for channels, shard_elements in work_items:
                    row_start = row * self.DRAM_column
                    if row_start >= shard_elements:
                        continue
                    elements = min(
                        self.DRAM_column,
                        shard_elements - row_start,
                    )
                    op_size = math.ceil(elements / self.burst_length)
                    self.WR_GB_only_trace(channels, op_size)
                    mac_channels_by_op_size.setdefault(op_size, []).extend(
                        channels
                    )
                for op_size, channels in mac_channels_by_op_size.items():
                    self.MAC_ABK_only_trace(
                        channels,
                        row_index_matrix + row,
                        op_size,
                        timing,
                    )
            self.RD_MAC_only_trace(active_channels)

    def Vector_Matrix_Mul_score_pim_only_trace(self, row_index_matrix, seqlen, timing):
        if self.kv_head_tp:
            self._Vector_Matrix_Mul_score_kv_head_tp_only_trace(
                row_index_matrix, seqlen, timing
            )
            return
        rows_per_vector = (self.head_dim * self.n_kv_heads - 1) // self.DRAM_column + 1
        MAC_op_size = self.head_dim // self.burst_length
        heads_per_row = self.DRAM_column // self.head_dim
        # channels_required = self.channels_per_block
        channels_required_all_devices = self.FC_total_banks // self.num_banks
        seq_iterations = (seqlen - 1) // self.FC_total_banks + 1
        for row_index in range(rows_per_vector):
            for seq_iter in range(seq_iterations):
                if seq_iter == seq_iterations - 1:
                    left_channels = (seqlen - self.FC_total_banks * seq_iter - 1) // self.num_banks + 1
                else:
                    left_channels = channels_required_all_devices
                channel_multi_transformer_block_required = 32 if channels_required_all_devices > 32 else self.num_channels // left_channels * left_channels
                channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
                for repeat in range(self.n_repeat):
                    heads_this_row = min(
                        heads_per_row,
                        self.n_kv_heads - row_index * heads_per_row,
                    )
                    WR_GB_op_size = heads_this_row * self.head_dim // self.burst_length
                    self.WR_GB_only_trace(channel_lst, WR_GB_op_size)
                    for head_iter in range(heads_this_row):
                        self.WR_BIAS_only_trace(channel_lst)
                        self.MAC_ABK_only_trace(channel_lst, row_index_matrix + seq_iter * rows_per_vector + row_index, MAC_op_size, timing)
                        self.RD_MAC_only_trace(channel_lst)
    
    def Vector_Matrix_Mul_output_pim_only_trace(self, row_index_matrix, seqlen, timing):
        if self.kv_head_tp:
            self._Vector_Matrix_Mul_output_kv_head_tp_only_trace(
                row_index_matrix, seqlen, timing
            )
            return
        if self.intra_device_attention:
            rows_per_seq = (seqlen - 1) // self.DRAM_column + 1
            rows_per_dim = (self.max_seq_len - 1) // self.DRAM_column + 1
            left_banks = self.num_banks
            num_heads_per_bank = (self.n_kv_heads - 1) // self.channels_per_block + 1
            channel_lst = [i for i in range(self.channels_per_block)]
            channel_multi_transformer_block_required = self.num_channels // self.channels_per_block * self.channels_per_block
            channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
            dim_iterations = self.head_dim // left_banks
            for head_index_per_bank in range(num_heads_per_bank):
                row_current_head = row_index_matrix + (rows_per_dim * dim_iterations) * head_index_per_bank
                for repeat in range(self.n_repeat):
                    for row_offset in range(rows_per_seq):
                        if row_offset == rows_per_seq - 1:
                            op_size = (seqlen - row_offset * self.DRAM_column - 1) // self.burst_length + 1
                        else:
                            op_size = self.DRAM_column // self.burst_length
                        self.WR_GB_only_trace(channel_lst, op_size)
                        for dim_iter in range(dim_iterations):
                            self.WR_BIAS_only_trace(channel_lst)
                            self.MAC_ABK_only_trace(channel_lst, row_current_head + dim_iter * rows_per_dim + row_offset, op_size, timing)
                            self.RD_MAC_only_trace(channel_lst)
        else:
            channels_required_all_devices = self.FC_total_banks // self.num_banks
            channel_multi_transformer_block_required = 32 if channels_required_all_devices > 32 else self.num_channels // channels_required_all_devices * channels_required_all_devices
            channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
            # if banks_per_head < 16, channels_per_head < 1, one bank has more than one head, throw error
            # if 16 <= banks_per_head < 128, dim_iterations > 1, channels_per_head >= 1
            # if 128 <= banks_per_head < 512, dim_iterations = 1, devices_per_head = 1
            # if 512 <= banks_per_head, dim_iterations = 1, devices_per_head > 1
                                                                                                # seqlen = 32k, head_dim = 128
            banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1                   # 8,  32, 256, 2k
            channels_per_head = (banks_per_head - 1) // (self.num_banks) + 1                    # 1,  2,  16,  128
            devices_per_head = (channels_per_head - 1) // (self.num_channels) + 1               # 1,  1,  1,   4
            # iteration along the head dimension
            dim_iterations = (self.head_dim - 1) // banks_per_head + 1                          # 16, 4,  1,   1
            # iteration along the sequence dimension or rows per sequence
            rows_per_seq_iteration = (banks_per_head - 1) // self.head_dim + 1                  # 1,  1,  2,   16
            seq_iterations = (seqlen - 1) // (self.DRAM_column * rows_per_seq_iteration) + 1    # 32, 32, 16,  2
            rows_per_seq = (seqlen - 1) // (self.DRAM_column) + 1                               # 32, 32, 32,  32
            channels_per_row_offset = (self.head_dim - 1) // self.num_banks + 1  

            if banks_per_head < self.num_banks:
                raise ValueError("banks_per_head < self.num_banks. One head is mapped to less than one channel. Not enough channels are allocated.")
            for repeat in range(self.n_repeat):
                for row_offset in range(rows_per_seq):
                    if row_offset == rows_per_seq - 1:
                        op_size = (seqlen - row_offset * self.DRAM_column - 1) // self.burst_length + 1
                    else:
                        op_size = self.DRAM_column // self.burst_length
                    self.WR_GB_only_trace(channel_lst, op_size)
                    if banks_per_head < 128:    # dim_iterations > 1, more than one dim in each row_offset are stored to a bank
                        for dim_iter in range(dim_iterations):   # E.g., head_dim = 128, banks_per_head = 32, channels_per_head = 2, dim_iterations = 128 / 32 = 4 in each bank. Within each iteration, Channel 0 is responsible for head 0: [0-15, 32-47, 64-79, 96-111], Channel 1 is responsible for head 1: [16-31, 48-63, 80-95, 112-127]. For bias vector, each head looks like (----CH0 16 Banks----,----CH1 16 Banks----) * 4.
                            self.WR_BIAS_only_trace(channel_lst)
                            self.MAC_ABK_only_trace(channel_lst, row_index_matrix + row_offset * dim_iterations + dim_iter, op_size, timing)
                            self.RD_MAC_only_trace(channel_lst)
                    else:
                        self.WR_BIAS_only_trace(channel_lst)
                        self.MAC_ABK_only_trace(channel_lst, row_index_matrix + row_offset // rows_per_seq_iteration, op_size, timing)
                        self.RD_MAC_only_trace(channel_lst)


    def Vector_Matrix_Mul_weight_systolic_pim(self, vector, row_index_matrix, vector_dim, matrix_col, FC_total_banks, op_trace_input):
        matrix_cols = matrix_col     # 1024: Wk in Llama3 70B
        matrix_rows = vector_dim     # 8192: Wk in Llama3 70B
        matrix_cols_per_bank = math.ceil(matrix_cols / self.FC_total_banks)
        banks_per_matrix_col = math.ceil(self.burst_length / matrix_cols_per_bank)
        # print("matrix_cols", matrix_cols, "matrix_rows", matrix_rows, "total banks", self.FC_total_banks, "matrix_cols_per_bank", matrix_cols_per_bank, "banks_per_matrix_col", banks_per_matrix_col)
        matrix_cols_per_bank = max(matrix_cols_per_bank, self.burst_length)
        bursts_of_matrix_cols_per_bank = math.ceil(matrix_cols_per_bank / self.burst_length)
        matrix_cols_per_bank_padding = bursts_of_matrix_cols_per_bank * self.burst_length  # round up to multiple of burst_length
        bursts_per_DRAM_row = self.DRAM_column // self.burst_length
        assert matrix_rows // banks_per_matrix_col * self.burst_length >= self.DRAM_column, "Each bank has more than one row of data"
        GQA = self.n_repeat
        systolic_dim_padding = self.systolic_dim
        chunk_size = self.DRAM_column // systolic_dim_padding # 1K / 8 = 128
        chunks = math.ceil(vector_dim / chunk_size)     # 4K / 128 = 32
        DRAM_rows_per_chunk = math.ceil(chunk_size / bursts_per_DRAM_row)
        assert vector.shape[0] == self.systolic_dim
        results = torch.zeros(torch.Size([self.systolic_dim, matrix_col]))

        if banks_per_matrix_col > 1:    # Partition matrix_rows into banks
            banks_per_matrix_row = matrix_cols // self.burst_length
            banks_per_matrix_col = self.FC_total_banks // banks_per_matrix_row
            utilized_banks = banks_per_matrix_row * banks_per_matrix_col
            matrix_rows_per_bank = math.ceil(matrix_rows / banks_per_matrix_col)
            bursts_per_bank = math.ceil(matrix_rows_per_bank * matrix_cols_per_bank_padding / self.burst_length)
            DRAM_rows_per_bank = math.ceil(bursts_per_bank / bursts_per_DRAM_row)
            print("utilized_banks", utilized_banks, "bursts_of_matrix_cols_per_bank", bursts_of_matrix_cols_per_bank, "matrix_cols_per_bank_padding", matrix_cols_per_bank_padding, "matrix_rows_per_bank", matrix_rows_per_bank, "banks_per_matrix_col", banks_per_matrix_col, "DRAM_rows_per_bank", DRAM_rows_per_bank, "chunk_size", chunk_size, "chunks", chunks)
            channels_utilized = self.channels_per_block
            channels_required_all_devices = (utilized_banks - 1) // self.num_banks + 1
            result_tmp_tensor = torch.zeros(torch.Size([banks_per_matrix_col, self.systolic_dim, matrix_col]))

            for channel in range(channels_required_all_devices):
                op_trace = channel == 0 and op_trace_input
                if channel == channels_required_all_devices - 1:
                    left_banks = utilized_banks - self.num_banks * (channels_required_all_devices - 1)
                else:
                    left_banks = self.num_banks

                matrix_row_block_index = channel * self.num_banks // banks_per_matrix_row
                if matrix_row_block_index == banks_per_matrix_col - 1:
                    matrix_rows_per_bank_per_row_block = matrix_rows - matrix_row_block_index * matrix_rows_per_bank
                else:
                    matrix_rows_per_bank_per_row_block = matrix_rows_per_bank
                chunks_per_row_block = math.ceil(matrix_rows_per_bank_per_row_block / chunk_size)

                self.WR_BIAS_systolic_pim(channel // self.num_channels, channel % self.num_channels, channels_utilized, [0 for bank in range(self.num_banks)], systolic_dim_padding, op_trace)

                for chunk_index in range(chunks_per_row_block):
                    if chunk_index == chunks_per_row_block - 1:
                        vector_size = matrix_rows_per_bank_per_row_block - chunk_index * chunk_size
                    else:
                        vector_size = chunk_size
                    DRAM_rows_this_chunk = math.ceil(vector_size / bursts_per_DRAM_row) # the last chunk may only have one DRAM row
                    vector_WR_GB = vector[:, matrix_row_block_index * matrix_rows_per_bank + chunk_index * chunk_size : matrix_row_block_index * matrix_rows_per_bank + chunk_index * chunk_size + vector_size].transpose(0, 1).reshape(-1)  # [8, 128] -> [128, 8] -> [1024], [43, 8] -> [344]
                    GB_op_size = math.ceil(vector_size * systolic_dim_padding / self.burst_length)
                    current_dim = vector_WR_GB.shape[0]
                    remainder = GB_op_size * self.burst_length - current_dim
                    if remainder != 0:
                        padding_needed = 16 - remainder
                        vector_WR_GB = F.pad(vector_WR_GB, (0, padding_needed), "constant", 0)
                    self.WR_GB_systolic_pim(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, GB_op_size, vector_WR_GB, op_trace)
                    for DRAM_row in range(DRAM_rows_this_chunk):  # not considering fragmentation
                        if DRAM_row == DRAM_rows_this_chunk - 1:
                            op_size = vector_size - DRAM_row * bursts_per_DRAM_row
                        else:
                            op_size = bursts_per_DRAM_row
                        row_offset = chunk_index * DRAM_rows_per_chunk + DRAM_row
                        self.MAC_systolic_pim(channel // self.num_channels, channel % self.num_channels, channels_utilized, row_index_matrix + row_offset, DRAM_row * bursts_per_DRAM_row * self.systolic_dim, op_size, systolic_dim_padding, op_trace, False)
                result_tensor = self.RD_MAC_systolic_pim(channel // self.num_channels, channel % self.num_channels, channels_utilized, systolic_dim_padding, op_trace)
                for bank in range(left_banks):
                    matrix_col_index = ((channel * self.num_banks + bank) * matrix_cols_per_bank_padding) % matrix_cols
                    result_tmp_tensor[matrix_row_block_index][:, matrix_col_index : matrix_col_index + self.burst_length] = result_tensor[bank]
            results = torch.sum(result_tmp_tensor, dim=0)

        else:   # Keep all matrix_rows in a single bank, >=1 burst per bank
            utilized_banks = math.ceil(matrix_cols / matrix_cols_per_bank_padding)
            matrix_rows_per_bank = math.ceil(matrix_rows / banks_per_matrix_col)
            banks_per_matrix_row = utilized_banks
            bursts_per_bank = math.ceil(matrix_rows_per_bank * matrix_cols_per_bank_padding / self.burst_length)
            DRAM_rows_per_bank = math.ceil(bursts_per_bank / bursts_per_DRAM_row)
            DRAM_rows_per_burst_block = math.ceil(matrix_rows_per_bank / bursts_per_DRAM_row)
            channels_utilized = self.channels_per_block
            channels_required_all_devices = (utilized_banks - 1) // self.num_banks + 1
            print("utilized_banks", utilized_banks, "bursts_of_matrix_cols_per_bank", bursts_of_matrix_cols_per_bank, "matrix_cols_per_bank_padding", matrix_cols_per_bank_padding, "matrix_rows_per_bank", matrix_rows_per_bank, "banks_per_matrix_col", banks_per_matrix_col, "DRAM_rows_per_bank", DRAM_rows_per_bank, "chunk_size", chunk_size, "chunks", chunks, "channels_required_all_devices", channels_required_all_devices, "bursts_of_matrix_cols_per_bank", bursts_of_matrix_cols_per_bank, "DRAM_rows_per_chunk", DRAM_rows_per_chunk, "DRAM_rows_per_burst_block", DRAM_rows_per_burst_block)

            for channel in range(channels_required_all_devices):
                op_trace = channel == 0 and op_trace_input
                if channel == channels_required_all_devices - 1:
                    left_banks = utilized_banks - self.num_banks * (channels_required_all_devices - 1)
                else:
                    left_banks = self.num_banks

                for burst in range(bursts_of_matrix_cols_per_bank):
                    self.WR_BIAS_systolic_pim(channel // self.num_channels, channel % self.num_channels, channels_utilized, [0 for bank in range(self.num_banks)], systolic_dim_padding, op_trace)
                    for chunk_index in range(chunks):
                        if chunk_index == chunks - 1:
                            vector_size = vector_dim - chunk_index * chunk_size
                        else:
                            vector_size = chunk_size
                        vector_WR_GB = vector[:, chunk_index * chunk_size : chunk_index * chunk_size + vector_size].transpose(0, 1).reshape(-1)  # [8, 128] -> [128, 8] -> [1024]
                        GB_op_size = math.ceil(vector_size * systolic_dim_padding / self.burst_length)
                        self.WR_GB_systolic_pim(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, GB_op_size, vector_WR_GB, op_trace)
                        for DRAM_row in range(DRAM_rows_per_chunk):  # not considering fragmentation
                            row_offset = burst * DRAM_rows_per_burst_block + chunk_index * DRAM_rows_per_chunk + DRAM_row
                            self.MAC_systolic_pim(channel // self.num_channels, channel % self.num_channels, channels_utilized, row_index_matrix + row_offset, DRAM_row * bursts_per_DRAM_row * self.systolic_dim, bursts_per_DRAM_row, systolic_dim_padding, op_trace, False)
                    result_tensor = self.RD_MAC_systolic_pim(channel // self.num_channels, channel % self.num_channels, channels_utilized, systolic_dim_padding, op_trace)
                    for bank in range(left_banks):
                        matrix_col_index = (channel * self.num_banks + bank) * matrix_cols_per_bank_padding + burst * self.burst_length
                        if matrix_col_index >= matrix_cols:
                            break
                        results[:, matrix_col_index : matrix_col_index + self.burst_length] = result_tensor[bank]

        return results

    def Vector_Matrix_Mul_weight_pim(self, vector, row_index_matrix, vector_dim, matrix_col, FC_total_banks, op_trace_input, timing):
        matrix_col_per_bank = (matrix_col - 1) // FC_total_banks + 1
        utilized_banks = (matrix_col - 1) // matrix_col_per_bank + 1  # shape = [4096, 11008]
        rows_per_vector = (vector_dim - 1) // self.DRAM_column + 1
        accumulator = [0 for _ in range(matrix_col_per_bank * utilized_banks)]
        channels_utilized = self.channels_per_block
        channels_required_all_devices = (utilized_banks - 1) // self.num_banks + 1
        # print(matrix_col_per_bank, utilized_banks, channels_utilized)
        for channel in range(channels_required_all_devices):
            op_trace = channel == 0 and op_trace_input
            if channel == channels_required_all_devices - 1:
                left_banks = utilized_banks - self.num_banks * (channels_required_all_devices - 1)
            else:
                left_banks = self.num_banks
            if self.GEMV_order == "no-reuse":
                # keep vector static in GB and frequently write BIAS registers
                for row_index in range(rows_per_vector):
                    if row_index < rows_per_vector - 1:
                        vector_tmp = vector[row_index * self.DRAM_column : (row_index+1) * self.DRAM_column]
                    else:
                        vector_tmp = vector[row_index * self.DRAM_column :]
                    op_size = (vector_tmp.shape[0] - 1) // self.burst_length + 1

                    self.WR_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_size, vector_tmp, op_trace)
                    for vector_index_per_bank in range(matrix_col_per_bank):
                        bias = [accumulator[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] for bank in range(left_banks)] + [0 for bank in range(self.num_banks - left_banks)]
                        self.WR_BIAS(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, bias, op_trace)
                        self.MAC_BK_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, row_index_matrix + vector_index_per_bank * rows_per_vector + row_index, 0, 0, op_size, op_trace, timing)
                        accumulator_loaded = self.RD_MAC(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_trace)
                        for bank in range(left_banks):
                            accumulator[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] = accumulator_loaded[bank]
            elif self.GEMV_order == "reuse-GB":
                # keep vector static in GB and frequently write BIAS registers
                # write BIAS registers multiple times, then MAC_BK_GB multiple times, finally RD_MAC multiple times, saving the mode switching time between manipulating BIAS register and MAC_BK_GB
                num_reuse_groups = (matrix_col_per_bank - 1) // self.reuse_size + 1
                reuse_group_size = (matrix_col_per_bank - 1) // num_reuse_groups + 1
                for row_index in range(rows_per_vector):
                    if row_index < rows_per_vector - 1:
                        vector_tmp = vector[row_index * self.DRAM_column : (row_index+1) * self.DRAM_column]
                    else:
                        vector_tmp = vector[row_index * self.DRAM_column :]
                    op_size = (vector_tmp.shape[0] - 1) // self.burst_length + 1
                    self.WR_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_size, vector_tmp, op_trace)
                    for reuse_group_index in range(num_reuse_groups):
                        if reuse_group_index < num_reuse_groups - 1:
                            num_left_maxtrix_col = reuse_group_size
                        else:
                            num_left_maxtrix_col = matrix_col_per_bank - reuse_group_size * (num_reuse_groups - 1)
                        for latch_index in range(num_left_maxtrix_col):
                            vector_index_per_bank = reuse_group_index * reuse_group_size + latch_index
                            bias = [accumulator[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] for bank in range(left_banks)] + [0 for bank in range(self.num_banks - left_banks)]
                            self.WR_BIAS(channel // self.num_channels, channel % self.num_channels, channels_utilized, latch_index, bias, op_trace)
                        for latch_index in range(num_left_maxtrix_col):
                            vector_index_per_bank = reuse_group_index * reuse_group_size + latch_index
                            self.MAC_BK_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, row_index_matrix + vector_index_per_bank * rows_per_vector + row_index, 0, latch_index, op_size, op_trace, timing)
                        for latch_index in range(num_left_maxtrix_col):
                            vector_index_per_bank = reuse_group_index * reuse_group_size + latch_index
                            accumulator_loaded = self.RD_MAC(channel // self.num_channels, channel % self.num_channels, channels_utilized, latch_index, op_trace)
                            for bank in range(left_banks):
                                accumulator[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] = accumulator_loaded[bank]
        # print(self.pim_device["dimm_" + str(0)].dimm["channel_" + str(31)].channel["bank_" + str(15)].latch[0])
        # print(self.pim_device["dimm_" + str(0)].dimm["channel_" + str(31)].channel["bank_" + str(15)].arrays[18])
        # print(self.pim_device["dimm_" + str(1)].dimm["channel_" + str(31)].channel["bank_" + str(15)].latch[0])
        # print(self.pim_device["dimm_" + str(1)].dimm["channel_" + str(31)].channel["bank_" + str(15)].arrays[18])
        return torch.tensor(accumulator[:matrix_col])
    
    def Vector_Matrix_Mul_weight_af_pim(self, vector, row_index_matrix, vector_dim, matrix_col, FC_total_banks, op_trace_input, timing):
        matrix_col_per_bank = (matrix_col - 1) // FC_total_banks + 1
        utilized_banks = (matrix_col - 1) // matrix_col_per_bank + 1  # shape = [4096, 11008]
        rows_per_vector = (vector_dim - 1) // self.DRAM_column + 1
        accumulator = [0 for _ in range(matrix_col_per_bank * utilized_banks)]
        accumulator_af = [0 for _ in range(matrix_col_per_bank * utilized_banks)]
        channels_utilized = self.channels_per_block
        channels_required_all_devices = (utilized_banks - 1) // self.num_banks + 1
        for channel in range(channels_required_all_devices):
            op_trace = channel == 0 and op_trace_input
            if channel == channels_required_all_devices - 1:
                left_banks = utilized_banks - self.num_banks * (channels_required_all_devices - 1)
            else:
                left_banks = self.num_banks
            if self.GEMV_order == "no-reuse":
                # keep vector static in GB and frequently write BIAS registers
                for row_index in range(rows_per_vector):
                    if row_index < rows_per_vector - 1:
                        vector_tmp = vector[row_index * self.DRAM_column : (row_index+1) * self.DRAM_column]
                    else:
                        vector_tmp = vector[row_index * self.DRAM_column :]
                    op_size = (vector_tmp.shape[0] - 1) // self.burst_length + 1
                    self.WR_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_size, vector_tmp, op_trace)
                    for vector_index_per_bank in range(matrix_col_per_bank):
                        bias = [accumulator[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] for bank in range(left_banks)] + [0 for bank in range(self.num_banks - left_banks)]
                        self.WR_BIAS(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, bias, op_trace)
                        self.MAC_BK_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, row_index_matrix + vector_index_per_bank * rows_per_vector + row_index, 0, 0, op_size, op_trace, timing)
                        if row_index == rows_per_vector - 1:
                            self.AF(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_trace)
                            accumulator_af_loaded = self.RD_AF(channel // self.num_channels, channel % self.num_channels, channels_utilized, op_trace)
                            for bank in range(left_banks):
                                accumulator_af[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] = accumulator_af_loaded[bank]
                        accumulator_loaded = self.RD_MAC(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_trace)
                        for bank in range(left_banks):
                            accumulator[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] = accumulator_loaded[bank]
            elif self.GEMV_order == "reuse-GB":
                # keep vector static in GB and frequently write BIAS registers
                # write BIAS registers multiple times, then MAC_BK_GB multiple times, finally RD_MAC multiple times, saving the mode switching time between manipulating BIAS register and MAC_BK_GB
                num_reuse_groups = (matrix_col_per_bank - 1) // (self.reuse_size // 2) + 1
                reuse_group_size = (matrix_col_per_bank - 1) // num_reuse_groups + 1
                for row_index in range(rows_per_vector):
                    if row_index < rows_per_vector - 1:
                        vector_tmp = vector[row_index * self.DRAM_column : (row_index+1) * self.DRAM_column]
                    else:
                        vector_tmp = vector[row_index * self.DRAM_column :]
                    op_size = (vector_tmp.shape[0] - 1) // self.burst_length + 1
                    self.WR_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_size, vector_tmp, op_trace)
                    for reuse_group_index in range(num_reuse_groups):
                        if reuse_group_index < num_reuse_groups - 1:
                            num_left_maxtrix_col = reuse_group_size
                        else:
                            num_left_maxtrix_col = matrix_col_per_bank - reuse_group_size * (num_reuse_groups - 1)
                        for latch_index in range(num_left_maxtrix_col):
                            vector_index_per_bank = reuse_group_index * reuse_group_size + latch_index
                            bias = [accumulator[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] for bank in range(left_banks)] + [0 for bank in range(self.num_banks - left_banks)]
                            self.WR_BIAS(channel // self.num_channels, channel % self.num_channels, channels_utilized, latch_index, bias, op_trace)
                        for latch_index in range(num_left_maxtrix_col):
                            vector_index_per_bank = reuse_group_index * reuse_group_size + latch_index
                            self.MAC_BK_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, row_index_matrix + vector_index_per_bank * rows_per_vector + row_index, 0, latch_index, op_size, op_trace, timing)
                        if row_index == rows_per_vector - 1:
                            for latch_index in range(num_left_maxtrix_col):
                                vector_index_per_bank = reuse_group_index * reuse_group_size + latch_index
                                self.AF(channel // self.num_channels, channel % self.num_channels, channels_utilized, latch_index, op_trace)
                                accumulator_af_loaded = self.RD_AF(channel // self.num_channels, channel % self.num_channels, channels_utilized, op_trace)
                                for bank in range(left_banks):
                                    accumulator_af[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] = accumulator_af_loaded[bank]
                        for latch_index in range(num_left_maxtrix_col):
                            vector_index_per_bank = reuse_group_index * reuse_group_size + latch_index
                            accumulator_loaded = self.RD_MAC(channel // self.num_channels, channel % self.num_channels, channels_utilized, latch_index, op_trace)
                            for bank in range(left_banks):
                                accumulator[(channel * self.num_banks + bank) * matrix_col_per_bank + vector_index_per_bank] = accumulator_loaded[bank]
        return (torch.tensor(accumulator), torch.tensor(accumulator_af))

    def Vector_Matrix_Mul_score_systolic_pim(self, vector, row_index_matrix, op_trace_input):

        seqlen = self.cache_k.shape[1]   # [8, 76, 32, 128]
        DRAM_rows_per_head = self.head_dim * self.burst_length // self.DRAM_column   # 2
        total_bursts = math.ceil(seqlen / self.burst_length)
        seq_iterations = math.ceil(total_bursts / self.FC_total_banks)
        bursts_per_DRAM_row = self.DRAM_column // self.burst_length
        GQA = self.n_repeat
        systolic_dim_padding = min(self.systolic_dim, GQA)
        channels_utilized = self.channels_per_block
        channels_required_all_devices = self.FC_total_banks // self.num_banks
        results = torch.zeros(torch.Size([self.batch_size, self.n_heads, seqlen]))
        for i in range(self.batch_size):
            for kv_head in range(self.n_kv_heads):
                for seq_iter in range(seq_iterations):
                    if seq_iter == seq_iterations - 1:
                        left_banks = math.ceil((seqlen - self.burst_length * self.FC_total_banks * seq_iter) / self.burst_length)
                        left_channels = math.ceil(left_banks / self.num_banks)
                    else:
                        left_channels = channels_required_all_devices
                    for channel in range(left_channels):
                        op_trace = channel == 0 and op_trace_input
                        vector_WR_GB = vector[i][kv_head * GQA : (kv_head+1) * GQA].transpose(0, 1).reshape(-1)  # [8, 128] -> [128, 8] -> [1024], [1, 128] -> [128, 1] -> [128]
                        GB_op_size = vector_WR_GB.shape[0] // self.burst_length
                        self.WR_GB_systolic_pim(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, GB_op_size, vector_WR_GB, op_trace)
                        if (seq_iter == seq_iterations - 1) and (channel == left_channels - 1):
                            left_banks = math.ceil((seqlen - self.burst_length * self.FC_total_banks * seq_iter - self.burst_length * self.num_banks * channel) / self.burst_length)
                        else:
                            left_banks = self.num_banks
                        self.WR_BIAS_systolic_pim(channel // self.num_channels, channel % self.num_channels, channels_utilized, [0 for bank in range(self.num_banks)], systolic_dim_padding, op_trace)
                        for DRAM_row in range(DRAM_rows_per_head):
                            row_offset = i * seq_iterations * self.n_kv_heads * DRAM_rows_per_head + seq_iter * self.n_kv_heads * DRAM_rows_per_head + kv_head * DRAM_rows_per_head + DRAM_row
                            GB_col_index = DRAM_row * bursts_per_DRAM_row * systolic_dim_padding
                            self.MAC_systolic_pim(channel // self.num_channels, channel % self.num_channels, channels_utilized, row_index_matrix + row_offset, GB_col_index, bursts_per_DRAM_row, systolic_dim_padding, op_trace, False)
                        result_tensor = self.RD_MAC_systolic_pim(channel // self.num_channels, channel % self.num_channels, channels_utilized, systolic_dim_padding, op_trace)    # [8, 16]
                        for bank in range(left_banks):
                            for q_head in range(GQA):
                                score_per_head = result_tensor[bank][q_head]
                                seq_index = (seq_iter * self.FC_total_banks + channel * self.num_banks + bank) * self.burst_length
                                if seq_index + self.burst_length > seqlen:
                                    score_len = seqlen - seq_index
                                else:
                                    score_len = self.burst_length
                                results[i][kv_head * GQA + q_head][seq_index : seq_index + score_len] = score_per_head[:score_len]
        return results.reshape((self.batch_size, self.n_heads, 1, seqlen))

    def Vector_Matrix_Mul_score_pim(self, row_index_vector, row_index_matrix, op_trace_input, timing):
        bsz, seqlen, _, _ = self.cache_k.shape
        rows_per_vector = (self.head_dim * self.n_kv_heads - 1) // self.DRAM_column + 1
        WR_GB_op_size = self.DRAM_column // self.burst_length
        MAC_op_size = self.head_dim // self.burst_length
        accumulator = torch.zeros(torch.Size([bsz, self.n_heads, 1, seqlen]))
        heads_per_row = self.DRAM_column // self.head_dim
        # channels_required = self.channels_per_block
        channels_utilized = self.channels_per_block
        channels_required_all_devices = self.FC_total_banks // self.num_banks
        seq_iterations = (seqlen - 1) // self.FC_total_banks + 1
        for row_index in range(rows_per_vector):
            for seq_iter in range(seq_iterations):
                if seq_iter == seq_iterations - 1:
                    left_channels = (seqlen - self.FC_total_banks * seq_iter - 1) // self.num_banks + 1
                else:
                    left_channels = channels_required_all_devices
                for channel in range(left_channels):
                    op_trace = channel == 0 and op_trace_input
                    if (seq_iter == seq_iterations - 1) and (channel == left_channels - 1):
                        left_banks = seqlen - self.FC_total_banks * seq_iter - self.num_banks * channel
                    else:
                        left_banks = self.num_banks
                    for repeat in range(self.n_repeat):
                        vector = self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, row_index * self.n_repeat + repeat, row_index_vector, 0, WR_GB_op_size * self.burst_length, False)
                        self.WR_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, WR_GB_op_size, vector, op_trace)
                        for head_iter in range(heads_per_row):
                            bias = [accumulator[0][row_index * heads_per_row + head_iter * self.n_repeat + repeat][0][seq_iter * self.FC_total_banks + channel * self.num_banks + bank] for bank in range(left_banks)] + [0 for bank in range(self.num_banks - left_banks)]
                            self.WR_BIAS(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, bias, op_trace)
                            self.MAC_BK_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, row_index_matrix + seq_iter * rows_per_vector + row_index, head_iter * self.head_dim, 0, MAC_op_size, op_trace, timing)
                            accumulator_loaded = self.RD_MAC(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_trace)
                            for bank in range(left_banks):
                                accumulator[0][row_index * heads_per_row + head_iter * self.n_repeat + repeat][0][seq_iter * self.FC_total_banks + channel * self.num_banks + bank] = accumulator_loaded[bank]
        return accumulator

    def Vector_Matrix_Mul_output_pim_systolic_pim(self, scores, row_index_matrix, op_trace_input):
        # score = [bsz, self.n_heads, seqlen], cache_v = [bsz, seqlen, self.n_kv_heads, self.head_dim]
        # shape = data.shape  # [8, 76, 32, 128]
        # constant
        # each channel has 16 banks, each heads require 8 banks, each channel handle 2 heads
        num_banks_per_head_dim = self.head_dim // self.burst_length         # 8
        heads_per_channel = self.num_banks // num_banks_per_head_dim        # 2
        num_bursts_per_DRAM_row = self.DRAM_column // self.burst_length     # 64
        # Variable
        GQA = self.n_repeat
        systolic_dim_padding = min(self.systolic_dim, GQA)
        # systolic_pim = 8, chunk_size = 1024 // 8 = 128
        # systolic_pim = 1, chunk_size = 1024 // 8 = 1024
        seqlen = self.cache_v.shape[1]
        chunk_size = self.DRAM_column // systolic_dim_padding
        chunks = (seqlen - 1) // chunk_size + 1
        # each channel has 16 banks, executing one KV head, if there num_channels < n_kv_heads, each channel has multiple KV heads, otherwise each channel has one KV head and chunks can be scaled down based on the number of channels
        pairs_of_kv_heads = self.n_kv_heads // heads_per_channel    # 4 in Llama-70B and 16 in Llama-7B
        if self.channels_per_block < pairs_of_kv_heads:
            paired_kv_head_per_channel = math.ceil(pairs_of_kv_heads / self.channels_per_block)
            seqlen_iterations = 1
        else:
            paired_kv_head_per_channel = 1
            seqlen_iterations = self.channels_per_block // pairs_of_kv_heads
        channels_per_seqlen_iteration = math.ceil(pairs_of_kv_heads / paired_kv_head_per_channel)
        channels_utilized = channels_per_seqlen_iteration * seqlen_iterations
        chunks_per_seqlen_iteration = math.ceil(chunks / seqlen_iterations)
        # print("pairs_of_kv_heads: ", pairs_of_kv_heads, "paired_kv_head_per_channel: ", paired_kv_head_per_channel, "seqlen_iterations: ", seqlen_iterations, "channels_per_seqlen_iteration: ", channels_per_seqlen_iteration, "chunks_per_seqlen_iteration: ", chunks_per_seqlen_iteration, "channels_utilized: ", channels_utilized)
        results = torch.zeros(torch.Size([self.batch_size, self.n_heads, self.head_dim]))
        for batch_index in range(self.batch_size):
            for channel_index in range(channels_utilized):
                op_trace = channel_index == 0 and op_trace_input
                dimm_index = channel_index // self.num_channels
                for paired_kv_head in range(paired_kv_head_per_channel):
                    paired_kv_head_index = (channel_index * paired_kv_head_per_channel) % pairs_of_kv_heads + paired_kv_head
                    head_tensor = torch.zeros(torch.Size([2*GQA, self.head_dim]))
                    for seqlen_iteration in range(seqlen_iterations):
                        if seqlen_iteration < seqlen_iterations - 1:
                            seqlen_per_iteration = chunks_per_seqlen_iteration * chunk_size
                        else:
                            seqlen_per_iteration = seqlen - (seqlen_iterations - 1) * chunks_per_seqlen_iteration * chunk_size
                        seqlen_iteration_tensor = torch.empty(torch.Size([2*GQA, self.head_dim]))
                        self.WR_BIAS_systolic_pim(dimm_index, channel_index % self.num_channels, channels_utilized, [0 for bank in range(self.num_banks)], systolic_dim_padding, op_trace)
                        for chunk in range(chunks_per_seqlen_iteration):
                            chunk_index = seqlen_iteration * chunks_per_seqlen_iteration + chunk
                            # burst_index_per_head = bank_index_per_iteration % bursts_per_head
                            if chunk < chunks_per_seqlen_iteration - 1:
                                seqlen_per_chunk = chunk_size
                            else:
                                if chunk_index >= chunks:
                                    break
                                seqlen_per_chunk = seqlen_per_iteration - (chunks_per_seqlen_iteration - 1) * chunk_size
                            rows_per_chunk = math.ceil(seqlen_per_chunk * self.burst_length / self.DRAM_column)
                            for row in range(rows_per_chunk):
                                if row == rows_per_chunk - 1:
                                    row_len = seqlen_per_chunk * self.burst_length - row * self.DRAM_column
                                else:
                                    row_len = self.DRAM_column
                                left_seqlen = row_len // self.burst_length
                                vector_WR_GB_a = scores[batch_index, 2*paired_kv_head_index * GQA : (2*paired_kv_head_index+1) * GQA, chunk_index * chunk_size + row * num_bursts_per_DRAM_row : chunk_index * chunk_size + row * num_bursts_per_DRAM_row + left_seqlen].transpose(0, 1).reshape(-1)  # [8, 64] -> [64, 8] -> [512]
                                vector_WR_GB_b = scores[batch_index, (2*paired_kv_head_index+1) * GQA : (2*paired_kv_head_index+2) * GQA, chunk_index * chunk_size + row * num_bursts_per_DRAM_row : chunk_index * chunk_size + row * num_bursts_per_DRAM_row + left_seqlen].transpose(0, 1).reshape(-1)  # [8, 64] -> [64, 8] -> [512]
                                vector_WR_GB = torch.concatenate((vector_WR_GB_a, vector_WR_GB_b), dim=0)
                                GB_op_size = math.ceil(vector_WR_GB.shape[0] / self.burst_length)
                                if vector_WR_GB.shape[0] < GB_op_size * self.burst_length:
                                    vector_WR_GB = torch.cat((vector_WR_GB, torch.zeros(GB_op_size * self.burst_length - vector_WR_GB.shape[0])))
                                self.WR_GB_systolic_pim(dimm_index, channel_index % self.num_channels, channels_utilized, 0, GB_op_size, vector_WR_GB, op_trace)
                                row_offset = batch_index * paired_kv_head_per_channel * seqlen_iterations * chunks_per_seqlen_iteration * rows_per_chunk + paired_kv_head * seqlen_iterations * chunks_per_seqlen_iteration * rows_per_chunk + seqlen_iteration * chunks_per_seqlen_iteration * rows_per_chunk + chunk * rows_per_chunk + row
                                # if channel_index == 0 and paired_kv_head == 0 and seqlen_iteration == 0 and chunk == 0 and row == 0:
                                #     flag = True
                                # else:
                                #     flag = False
                                self.MAC_output_systolic_pim(dimm_index, channel_index % self.num_channels, channels_utilized, row_index_matrix + row_offset, 0, left_seqlen, systolic_dim_padding, op_trace, False)
                        result_tensor = self.RD_MAC_systolic_pim(dimm_index, channel_index % self.num_channels, channels_utilized, systolic_dim_padding, op_trace)
                        for bank in range(self.num_banks):
                            for q_head in range(GQA):
                                seqlen_iteration_tensor[0*GQA + q_head, bank * self.burst_length // 2: (bank+1) * self.burst_length // 2] = result_tensor[bank, q_head, :self.burst_length // 2]
                                seqlen_iteration_tensor[1*GQA + q_head, bank * self.burst_length // 2: (bank+1) * self.burst_length // 2] = result_tensor[bank, q_head, self.burst_length // 2:]
                        head_tensor += seqlen_iteration_tensor
                    results[batch_index, 2*paired_kv_head_index * GQA : (2*paired_kv_head_index+1) * GQA] = head_tensor[0: GQA]
                    results[batch_index, (2*paired_kv_head_index+1) * GQA : (2*paired_kv_head_index+2) * GQA] = head_tensor[GQA: 2*GQA]
        return results.reshape((self.batch_size, 1, self.n_heads * self.head_dim))


    def Vector_Matrix_Mul_output_pim(self, scores, row_index_matrix, op_trace_input, timing):
        bsz, seqlen, _, _ = self.cache_k.shape
        accumulator = torch.zeros(torch.Size([bsz, self.n_heads, 1, self.head_dim]))
        if self.intra_device_attention:
            channels_required = self.channels_per_block
            rows_per_seq = (seqlen - 1) // self.DRAM_column + 1
            rows_per_dim = (self.max_seq_len - 1) // self.DRAM_column + 1
            num_heads_per_bank = (self.n_kv_heads - 1) // self.channels_per_block + 1
            for channel in range(channels_required):
                op_trace = channel == 0 and op_trace_input
                if channel == channels_required - 1:
                    num_heads_iteration = self.n_kv_heads - num_heads_per_bank * (self.channels_per_block - 1)
                else:
                    num_heads_iteration = num_heads_per_bank
                left_banks = self.num_banks
                dim_iterations = self.head_dim // left_banks
                for head_index_per_bank in range(num_heads_iteration):     # each head is distributed into all banks in a channel, each bank contains left_banks heads
                    row_current_head = row_index_matrix + (rows_per_dim * dim_iterations) * head_index_per_bank
                    for repeat in range(self.n_repeat):
                        head = channel * num_heads_per_bank + head_index_per_bank
                        if head > self.n_kv_heads - 1:
                            break
                        for row_offset in range(rows_per_seq):
                            if row_offset == rows_per_seq - 1:
                                op_size = (seqlen - row_offset * self.DRAM_column - 1) // self.burst_length + 1
                                pad_zero = torch.tensor([0 for _ in range(op_size * self.burst_length - (seqlen - row_offset * self.DRAM_column))])
                                vector = scores[0][head * self.n_repeat + repeat][0][row_offset * self.DRAM_column :]
                                vector = torch.cat((vector, pad_zero))
                            else:
                                op_size = self.DRAM_column // self.burst_length
                                vector = scores[0][head * self.n_repeat + repeat][0][row_offset * self.DRAM_column : (row_offset+1) * self.DRAM_column]
                            # vector = self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, head_index_per_bank * self.n_repeat + repeat, row_index_vector + row_offset, 0, op_size * self.burst_length, False)
                            self.WR_GB(channel // self.num_channels, channel % self.num_channels, channels_required, 0, op_size, vector, op_trace)
                            for dim_iter in range(dim_iterations):   # each head has dim 128, but distributed to 16 banks, so has 8 iterations in each bank
                                bias = [accumulator[0][head * self.n_repeat + repeat][0][dim_iter * left_banks + bank] for bank in range(left_banks)] + [0 for bank in range(self.num_banks - left_banks)]
                                self.WR_BIAS(channel // self.num_channels, channel % self.num_channels, channels_required, 0, bias, op_trace)
                                self.MAC_BK_GB(channel // self.num_channels, channel % self.num_channels, channels_required, row_current_head + dim_iter * rows_per_dim + row_offset, 0, 0, op_size, op_trace, timing)
                                accumulator_loaded = self.RD_MAC(channel // self.num_channels, channel % self.num_channels, channels_required, 0, op_trace)
                                for bank in range(left_banks):
                                    accumulator[0][head * self.n_repeat + repeat][0][dim_iter * left_banks + bank] = accumulator_loaded[bank]
        else:
            channels_utilized = self.channels_per_block
            channels_required_all_devices = self.FC_total_banks // self.num_banks
            # if banks_per_head < 16, channels_per_head < 1, one bank has more than one head, throw error
            # if 16 <= banks_per_head < 128, dim_iterations > 1, channels_per_head >= 1
            # if 128 <= banks_per_head < 512, dim_iterations = 1, devices_per_head = 1
            # if 512 <= banks_per_head, dim_iterations = 1, devices_per_head > 1
                                                                                                # seqlen = 32k, head_dim = 128
            banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1                   # 8,  32, 256, 2k
            channels_per_head = (banks_per_head - 1) // (self.num_banks) + 1                    # 1,  2,  16,  128
            devices_per_head = (channels_per_head - 1) // (self.num_channels) + 1               # 1,  1,  1,   4
            # iteration along the head dimension
            dim_iterations = (self.head_dim - 1) // banks_per_head + 1                          # 16, 4,  1,   1
            # iteration along the sequence dimension or rows per sequence
            rows_per_seq_iteration = (banks_per_head - 1) // self.head_dim + 1                  # 1,  1,  2,   16
            seq_iterations = (seqlen - 1) // (self.DRAM_column * rows_per_seq_iteration) + 1    # 32, 32, 16,  2
            rows_per_seq = (seqlen - 1) // (self.DRAM_column) + 1                               # 32, 32, 32,  32
            channels_per_row_offset = (self.head_dim - 1) // self.num_banks + 1                 # 8
            for channel in range(channels_required_all_devices):
                op_trace = channel == 0 and op_trace_input
                if banks_per_head < self.num_banks:
                    raise ValueError("banks_per_head < self.num_banks. One head is mapped to less than one channel. Not enough channels are allocated.")
                for repeat in range(self.n_repeat):
                    head = channel // (banks_per_head // self.num_banks)
                    print("repeat", repeat, "kv head", head, "head", head * self.n_repeat + repeat)
                    for row_offset in range(rows_per_seq):
                        if row_offset == rows_per_seq - 1:
                            op_size = (seqlen - row_offset * self.DRAM_column - 1) // self.burst_length + 1
                            pad_zero = torch.tensor([0 for _ in range(op_size * self.burst_length - (seqlen - row_offset * self.DRAM_column))])
                            vector = scores[0][head * self.n_repeat + repeat][0][row_offset * self.DRAM_column :]
                            vector = torch.cat((vector, pad_zero))
                        else:
                            op_size = self.DRAM_column // self.burst_length
                            vector = scores[0][head * self.n_repeat + repeat][0][row_offset * self.DRAM_column : (row_offset+1) * self.DRAM_column]
                        self.WR_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_size, vector, op_trace)
                        if banks_per_head < 128:    # dim_iterations > 1, more than one dim in each row_offset are stored to a bank
                            for dim_iter in range(dim_iterations):   # E.g., head_dim = 128, banks_per_head = 32, channels_per_head = 2, dim_iterations = 128 / 32 = 4 in each bank. Within each iteration, Channel 0 is responsible for head 0: [0-15, 32-47, 64-79, 96-111], Channel 1 is responsible for head 1: [16-31, 48-63, 80-95, 112-127]. For bias vector, each head looks like (----CH0 16 Banks----,----CH1 16 Banks----) * 4.
                                bias = [accumulator[0][head * self.n_repeat + repeat][0][dim_iter * banks_per_head + (channel % channels_per_head) * self.num_banks + bank] for bank in range(self.num_banks)]
                                self.WR_BIAS(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, bias, op_trace)
                                self.MAC_BK_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, row_index_matrix + row_offset * dim_iterations + dim_iter, 0, 0, op_size, op_trace, timing)
                                accumulator_loaded = self.RD_MAC(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_trace)
                                for bank in range(self.num_banks):
                                    accumulator[0][head * self.n_repeat + repeat][0][dim_iter * banks_per_head + (channel % channels_per_head) * self.num_banks + bank] = accumulator_loaded[bank]
                        else:   
                            # each head is mapped on a single device, channels_per_row_offset = 128 / 16 = 8
                            # E.g., head_dim = 128, banks_per_head = 256, channels_per_head = 16. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 7 is responsible for head 0: [0][112-127], Channel 8 is responsible for head 0: [1][0-15], Channel 9 is responsible for head 0: [1][16-31], ..., Channel 15 is responsible for head 0: [1][112-127]. For bias vector, each head has rows_per_seq_iteration = 2: (----CH0 16 Banks----) * 8, (----CH8 16 Banks----) * 8.
                            # each head is mapped on multiple devices
                            # E.g. head_dim = 128, banks_per_head = 2048, channels_per_head = 128. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 127 is responsible for head 0: [15][112-127]. For bias vector, each head has rows_per_seq_iteration = 16: (----CH0 16 Banks----) * 128, ..., (----CH112 16 Banks----) * 128.
                            if channel // channels_per_row_offset == row_offset:    
                                bias = [accumulator[0][head * self.n_repeat + repeat][0][((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank] for bank in range(self.num_banks)]
                                self.WR_BIAS(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, bias, op_trace)
                                self.MAC_BK_GB(channel // self.num_channels, channel % self.num_channels, channels_utilized, row_index_matrix + row_offset // rows_per_seq_iteration, 0, 0, op_size, op_trace, timing)
                                accumulator_loaded = self.RD_MAC(channel // self.num_channels, channel % self.num_channels, channels_utilized, 0, op_trace)
                                for bank in range(self.num_banks):
                                    accumulator[0][head * self.n_repeat + repeat][0][((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank] = accumulator_loaded[bank]
                    if head > 0:
                        print(vector)
        return accumulator
    
    def store_for_neighbor_bank_input_only_trace(self, channels_required, total_banks, bank_group_index, row_index, size):
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)
        for i in range((size - 1) // self.burst_length + 1):
            for bank in range(total_banks):
                dimm_index, channel_index, bank_index = self.bank_index(bank*2+bank_group_index)
                for tb in range(num_transformer_blocks_per_device):
                    self.W_MEM_only_trace(channel_index + channels_required * tb, bank_index, row_index, self.burst_length)
    
    def store_for_input_only_trace(self, channels_required, total_banks, row_index, size):
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)
        for i in range((size - 1) // self.burst_length + 1):
            for bank in range(total_banks):
                dimm_index, channel_index, bank_index = self.bank_index(bank)
                for tb in range(num_transformer_blocks_per_device):
                    self.W_MEM_only_trace(channel_index + channels_required * tb, bank_index, row_index, self.burst_length)

    
    def store_for_EWMUL_input_only_trace(self, channels_required, total_banks, bank_group_index, row_index, size):
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)
        for i in range((size - 1) // self.burst_length + 1):
            for bank in range(total_banks):
                dimm_index, channel_index, bank_index = self.bank_index(bank*4+bank_group_index)
                for tb in range(num_transformer_blocks_per_device):
                    self.W_MEM_only_trace(channel_index + channels_required * tb, bank_index, row_index, self.burst_length)
    
    def load_from_input_only_trace(self, channels_required, total_banks, row_index, size):
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)
        for i in range((size - 1) // self.burst_length + 1):
            for bank in range(total_banks):
                dimm_index, channel_index, bank_index = self.bank_index(bank)
                for tb in range(num_transformer_blocks_per_device):
                    self.R_MEM_only_trace(channel_index + channels_required * tb, bank_index, row_index, self.burst_length)
    
    def load_from_EWMUL_input_only_trace(self, channels_required, total_banks, bank_group_index, row_index, size):
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)
        for i in range((size - 1) // self.burst_length + 1):
            for bank in range(total_banks):
                dimm_index, channel_index, bank_index = self.bank_index(bank*4+bank_group_index)
                for tb in range(num_transformer_blocks_per_device):
                    self.R_MEM_only_trace(channel_index + channels_required * tb, bank_index, row_index, self.burst_length)
    
    
    def store_for_EWMUL_score_only_trace(self, channels_required, row_index, total_banks, bank_group_index, seqlen):
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)
        # size_per_bank = (self.n_heads * seqlen - 1) // (total_banks // 4) + 1
        # rows_per_bank = (size_per_bank - 1) // self.DRAM_column + 1
        rows_per_score = (seqlen - 1) // self.DRAM_column + 1
        num_heads_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
        for k in range(rows_per_score):
            if k == rows_per_score - 1:
                # size = (size_per_bank - 1) % self.DRAM_column + 1
                size = seqlen - self.DRAM_column * (rows_per_score - 1)
            else:
                size = self.DRAM_column
            for j in range(num_heads_per_bank):
                for i in range((size - 1) // self.burst_length + 1):
                    for bank in range(total_banks):
                        if bank % 4 == bank_group_index:
                            head = (bank // 4) * num_heads_per_bank + j
                            if head > self.n_heads - 1:
                                break
                            dimm_index, channel_index, bank_index = self.bank_index(bank)
                            for tb in range(num_transformer_blocks_per_device):
                                self.W_MEM_only_trace(channel_index + channels_required * tb, bank_index, row_index + j * rows_per_score + k, self.burst_length)  
    
    def load_from_EWMUL_score_only_trace(self, channels_required, row_index, total_banks, bank_group_index, seqlen):
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)
        # size_per_bank = (self.n_heads * seqlen - 1) // (total_banks // 4) + 1
        # rows_per_bank = (size_per_bank - 1) // self.DRAM_column + 1
        rows_per_score = (seqlen - 1) // self.DRAM_column + 1
        num_heads_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
        for k in range(rows_per_score):
            if k == rows_per_score - 1:
                # size = (size_per_bank - 1) % self.DRAM_column + 1
                size = seqlen - self.DRAM_column * (rows_per_score - 1)
            else:
                size = self.DRAM_column
            for j in range(num_heads_per_bank):
                for i in range((size - 1) // self.burst_length + 1):
                    for bank in range(total_banks):
                        if bank % 4 == bank_group_index:
                            head = (bank // 4) * num_heads_per_bank + j
                            if head > self.n_heads - 1:
                                break
                            dimm_index, channel_index, bank_index = self.bank_index(bank)
                            for tb in range(num_transformer_blocks_per_device):
                                self.R_MEM_only_trace(channel_index + channels_required * tb, bank_index, row_index + j * rows_per_score + k, self.burst_length)

    def store_for_score_only_trace(self, row_index, FC_total_banks, seqlen):
        channels_required_all_devices = self.FC_total_banks // self.num_banks
        num_transformer_blocks_per_device = max(self.num_channels // channels_required_all_devices, 1)
        rows = max(1, (seqlen * self.n_heads - 1) // self.DRAM_column // self.num_banks // channels_required_all_devices + 1)
        columns = (seqlen * self.n_heads - 1) // rows // self.num_banks // channels_required_all_devices + 1
        # print(rows, columns, num_transformer_blocks_per_device)
        for row in range(rows):
            for burst in range((columns - 1) // self.burst_length + 1):
                for bank_index in range(self.num_banks):
                    for channel_index in range(channels_required_all_devices):
                        if channel_index > self.num_channels - 1:
                            break
                        for tb in range(num_transformer_blocks_per_device):
                            self.W_MEM_only_trace(channel_index + channels_required_all_devices * tb, bank_index, row_index + row * self.DRAM_column, self.burst_length)

    def load_for_score_only_trace(self, row_index, FC_total_banks, seqlen):
        channels_required_all_devices = self.FC_total_banks // self.num_banks
        num_transformer_blocks_per_device = max(self.num_channels // channels_required_all_devices, 1)
        rows = max(1, (seqlen * self.n_heads - 1) // self.DRAM_column // self.num_banks // channels_required_all_devices + 1)
        columns = (seqlen * self.n_heads - 1) // rows // self.num_banks // channels_required_all_devices + 1
        for row in range(rows):
            for burst in range((columns - 1) // self.burst_length + 1):
                for bank_index in range(self.num_banks):
                    for channel_index in range(channels_required_all_devices):
                        if channel_index > self.num_channels - 1:
                            break
                        for tb in range(num_transformer_blocks_per_device):
                            self.R_MEM_only_trace(channel_index + channels_required_all_devices * tb, bank_index, row_index + row * self.DRAM_column, self.burst_length)

    def memory_mapping(self):
        """Hook for trace-only helpers that do not allocate model tensors."""
        pass
