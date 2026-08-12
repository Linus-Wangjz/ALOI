"""Shape and physical-layout helpers for CENT's KV-head TP mapping.

The mapping deliberately keeps every KV head and its complete context on one
TP rank.  K is striped over banks along the sequence dimension.  V is striped
over dimensions while there are at most 128 banks per KV head.  Once a head
has more than 128 banks, its channels are split into groups of exactly 128
banks: every group stores all 128 dimensions for a disjoint context slice.
This lets each group broadcast its own score slice through the existing shared
GB and use MAC_ABK; no bank-private score operand is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


KV_HEAD_TP_VALUES = (1, 2, 4, 8)
KV_HEAD_TP_ROLES = ("main", "helper")
SYSTOLIC_SV_PACKING_MODES = (
    "paired_kv_heads",
    "paired_query_groups",
    "paired_context_shards",
)


@dataclass(frozen=True)
class KVHeadTPShape:
    tp: int
    dim: int
    head_dim: int
    global_query_heads: int
    global_kv_heads: int
    global_ffn_dim: int
    local_query_heads: int
    local_kv_heads: int
    local_query_dim: int
    local_kv_dim: int
    local_ffn_dim: int

    @property
    def gqa_factor(self) -> int:
        return self.local_query_heads // self.local_kv_heads


@dataclass(frozen=True)
class SystolicProjectionLayout:
    """Physical placement of one logical projection on a TP rank."""

    name: str
    input_dim: int
    output_dim: int
    channels_per_group: int
    reduction_groups: int
    reduction_slice: int
    output_tiles: int = 1
    active_channels: int = 0
    fused_operands: int = 1

    @property
    def pnm_reduction_adds(self) -> int:
        return max(0, self.reduction_groups - 1) * self.output_dim


@dataclass(frozen=True)
class SystolicSVLayout:
    """Dual-half V-cache packing used by the systolic SV kernel."""

    mode: str
    channels_per_work_item: int
    channels_per_context_group: int
    context_groups_per_kv_head: int
    contexts_per_half: int
    query_waves: int
    active_systolic_rows: int
    active_lanes_per_bank: int
    systolic_height: int
    systolic_width: int

    @property
    def height_utilization(self) -> float:
        return self.active_systolic_rows / self.systolic_height

    @property
    def width_utilization(self) -> float:
        return self.active_lanes_per_bank / self.systolic_width


@dataclass(frozen=True)
class SystolicTPLayout:
    """Device/channel/bank layout for standard TP on systolic PIM.

    The layout assumes that the array width equals one DRAM burst.  Q and
    fused K/V are column-parallel projections whose reduction dimension may
    be split into independent channel groups.  Wo and W2 are row-parallel and
    always produce a full hidden-dimension partial vector.  W1/W3 are fused
    and tiled over output columns instead of splitting their reduction
    dimension.
    """

    shape: KVHeadTPShape
    num_channels: int
    banks_per_channel: int
    max_seq_len: int
    systolic_height: int
    dram_columns: int = 1024
    burst_length: int = 16

    def __post_init__(self) -> None:
        values = {
            "num_channels": self.num_channels,
            "banks_per_channel": self.banks_per_channel,
            "max_seq_len": self.max_seq_len,
            "systolic_height": self.systolic_height,
            "dram_columns": self.dram_columns,
            "burst_length": self.burst_length,
        }
        invalid = [name for name, value in values.items() if value <= 0]
        if invalid:
            raise ValueError(
                "Systolic TP layout values must be positive: " + ", ".join(invalid)
            )
        if self.dram_columns % self.burst_length:
            raise ValueError("DRAM row length must be a multiple of burst length")
        if self.burst_length % 2:
            raise ValueError("dual-half SV packing requires an even burst length")
        if self.total_banks % self.shape.local_kv_heads:
            raise ValueError("device banks must divide evenly across local KV heads")
        if self.num_channels % self.shape.local_kv_heads:
            raise ValueError("device channels must divide evenly across local KV heads")
        for projection in (self.q, self.kv):
            if projection.channels_per_group > self.num_channels:
                raise ValueError(
                    f"{projection.name} output columns do not fit one device group"
                )
            if self.num_channels % projection.channels_per_group:
                raise ValueError(
                    f"{projection.name} channel groups must divide the device evenly"
                )
        if self.shape.head_dim % (self.banks_per_channel * self.half_width):
            raise ValueError(
                "dual-half SV packing requires an integer number of channels to "
                "expose one head_dim across their half-width bank lanes"
            )
        if self.num_channels % self.channels_per_sv_group:
            raise ValueError("SV channel groups must divide the device evenly")

    @property
    def total_banks(self) -> int:
        return self.num_channels * self.banks_per_channel

    @property
    def half_width(self) -> int:
        return self.burst_length // 2

    @property
    def output_columns_per_channel(self) -> int:
        return self.banks_per_channel * self.burst_length

    @property
    def channels_per_sv_group(self) -> int:
        """Physical channels needed to expose one 128-element V vector half."""

        return self.shape.head_dim // (self.banks_per_channel * self.half_width)

    def _row_parallel_projection(
        self, name: str, input_dim: int
    ) -> SystolicProjectionLayout:
        device_output_capacity = self.total_banks * self.burst_length
        output_tiles = math.ceil(self.shape.dim / device_output_capacity)
        final_tile_outputs = (
            self.shape.dim - (output_tiles - 1) * device_output_capacity
        )
        final_tile_channels = math.ceil(
            final_tile_outputs / self.output_columns_per_channel
        )
        return SystolicProjectionLayout(
            name=name,
            input_dim=input_dim,
            output_dim=self.shape.dim,
            channels_per_group=self.num_channels,
            reduction_groups=1,
            reduction_slice=input_dim,
            output_tiles=output_tiles,
            active_channels=(
                self.num_channels if output_tiles > 1 else final_tile_channels
            ),
        )

    def _reduction_split_projection(
        self, name: str, input_dim: int, output_dim: int, fused_operands: int = 1
    ) -> SystolicProjectionLayout:
        device_output_capacity = self.total_banks * self.burst_length
        output_tiles = math.ceil(output_dim / device_output_capacity)
        tile_output = min(output_dim, device_output_capacity)
        channels_per_group = math.ceil(
            tile_output / self.output_columns_per_channel
        )
        reduction_groups = self.num_channels // channels_per_group
        return SystolicProjectionLayout(
            name=name,
            input_dim=input_dim,
            output_dim=output_dim,
            channels_per_group=channels_per_group,
            reduction_groups=reduction_groups,
            reduction_slice=math.ceil(input_dim / reduction_groups),
            output_tiles=output_tiles,
            active_channels=channels_per_group * reduction_groups,
            fused_operands=fused_operands,
        )

    @property
    def q(self) -> SystolicProjectionLayout:
        return self._reduction_split_projection(
            "q", self.shape.dim, self.shape.local_query_dim
        )

    @property
    def kv(self) -> SystolicProjectionLayout:
        return self._reduction_split_projection(
            "fused_kv", self.shape.dim, 2 * self.shape.local_kv_dim, 2
        )

    @property
    def wo(self) -> SystolicProjectionLayout:
        return self._row_parallel_projection("wo", self.shape.local_query_dim)

    @property
    def fused_ffn(self) -> SystolicProjectionLayout:
        combined_output = 2 * self.shape.local_ffn_dim
        device_output_capacity = self.total_banks * self.burst_length
        output_tiles = math.ceil(combined_output / device_output_capacity)
        final_tile_outputs = combined_output - (output_tiles - 1) * device_output_capacity
        final_tile_channels = math.ceil(
            min(device_output_capacity, final_tile_outputs)
            / self.output_columns_per_channel
        )
        return SystolicProjectionLayout(
            name="fused_w1_w3",
            input_dim=self.shape.dim,
            output_dim=combined_output,
            channels_per_group=self.num_channels,
            reduction_groups=1,
            reduction_slice=self.shape.dim,
            output_tiles=output_tiles,
            active_channels=(
                self.num_channels if output_tiles > 1 else final_tile_channels
            ),
            fused_operands=2,
        )

    @property
    def w2(self) -> SystolicProjectionLayout:
        return self._row_parallel_projection("w2", self.shape.local_ffn_dim)

    @property
    def channels_per_kv_head(self) -> int:
        return self.num_channels // self.shape.local_kv_heads

    @property
    def banks_per_kv_head(self) -> int:
        return self.total_banks // self.shape.local_kv_heads

    def k_contexts_per_bank(self, sequence_length: int) -> int:
        self._validate_sequence_length(sequence_length)
        return math.ceil(sequence_length / self.banks_per_kv_head)

    def k_width_utilization(self, sequence_length: int) -> float:
        return min(1.0, self.k_contexts_per_bank(sequence_length) / self.burst_length)

    @property
    def qk_query_waves(self) -> int:
        return math.ceil(self.shape.gqa_factor / self.systolic_height)

    @property
    def qk_height_utilization(self) -> float:
        return min(1.0, self.shape.gqa_factor / self.systolic_height)

    @property
    def sv_packing_mode(self) -> str:
        if self.shape.local_kv_heads >= 2 and self.shape.local_kv_heads % 2 == 0:
            return "paired_kv_heads"
        if 2 * self.systolic_height <= self.shape.gqa_factor:
            return "paired_query_groups"
        return "paired_context_shards"

    def sv(self, sequence_length: int) -> SystolicSVLayout:
        self._validate_sequence_length(sequence_length)
        mode = self.sv_packing_mode
        physical_group_channels = self.channels_per_sv_group
        if mode == "paired_kv_heads":
            work_items = self.shape.local_kv_heads // 2
            channels_per_work_item = self.num_channels // work_items
            if channels_per_work_item % physical_group_channels:
                raise ValueError("paired KV-head channel allocation is not group aligned")
            context_groups = channels_per_work_item // physical_group_channels
            contexts_per_half = math.ceil(sequence_length / context_groups)
            query_waves = math.ceil(self.shape.gqa_factor / self.systolic_height)
            active_rows = min(self.systolic_height, self.shape.gqa_factor)
        elif mode == "paired_query_groups":
            channels_per_work_item = self.num_channels
            context_groups = self.num_channels // physical_group_channels
            contexts_per_half = math.ceil(sequence_length / context_groups)
            query_waves = math.ceil(
                self.shape.gqa_factor / (2 * self.systolic_height)
            )
            active_rows = min(
                self.systolic_height,
                math.ceil(self.shape.gqa_factor / 2),
            )
        else:
            channels_per_work_item = self.num_channels
            context_groups = 2 * (
                self.num_channels // physical_group_channels
            )
            contexts_per_half = math.ceil(sequence_length / context_groups)
            query_waves = math.ceil(self.shape.gqa_factor / self.systolic_height)
            active_rows = min(self.systolic_height, self.shape.gqa_factor)
        return SystolicSVLayout(
            mode=mode,
            channels_per_work_item=channels_per_work_item,
            channels_per_context_group=physical_group_channels,
            context_groups_per_kv_head=context_groups,
            contexts_per_half=contexts_per_half,
            query_waves=query_waves,
            active_systolic_rows=active_rows,
            active_lanes_per_bank=self.burst_length,
            systolic_height=self.systolic_height,
            systolic_width=self.burst_length,
        )

    def sv_reduction_adds(self, sequence_length: int) -> int:
        groups = self.sv(sequence_length).context_groups_per_kv_head
        return self.shape.local_query_heads * self.shape.head_dim * max(0, groups - 1)

    def pnm_reduction_adds(self, sequence_length: int) -> dict[str, int]:
        return {
            "q": self.q.pnm_reduction_adds,
            "kv": self.kv.pnm_reduction_adds,
            "sv": self.sv_reduction_adds(sequence_length),
            "fused_ffn": self.fused_ffn.pnm_reduction_adds,
        }

    def _validate_sequence_length(self, sequence_length: int) -> None:
        if not 0 < sequence_length <= self.max_seq_len:
            raise ValueError(
                f"active sequence length must be in [1, {self.max_seq_len}]"
            )


@dataclass(frozen=True)
class KVHeadTPLayout:
    """Device/channel/bank layout for one symmetric KV-head TP rank."""

    shape: KVHeadTPShape
    num_channels: int
    banks_per_channel: int
    max_seq_len: int
    dram_columns: int = 1024
    burst_length: int = 16
    k_contexts_per_row: int = 8

    def __post_init__(self) -> None:
        values = {
            "num_channels": self.num_channels,
            "banks_per_channel": self.banks_per_channel,
            "max_seq_len": self.max_seq_len,
            "dram_columns": self.dram_columns,
            "burst_length": self.burst_length,
            "k_contexts_per_row": self.k_contexts_per_row,
        }
        invalid = [name for name, value in values.items() if value <= 0]
        if invalid:
            raise ValueError(f"KV-head TP layout values must be positive: {', '.join(invalid)}")
        if self.total_banks % self.shape.local_kv_heads:
            raise ValueError("device banks must divide evenly across local KV heads")
        if self.num_channels % self.shape.local_kv_heads:
            raise ValueError("device channels must divide evenly across local KV heads")
        if self.shape.head_dim % self.channels_per_kv_head:
            raise ValueError("head_dim must divide evenly across a KV head's channels")
        if self.v_context_split:
            if self.banks_per_kv_head % self.shape.head_dim:
                raise ValueError(
                    "banks per KV head must be a multiple of head_dim for V context shards"
                )
            if self.shape.head_dim % self.banks_per_channel:
                raise ValueError(
                    "head_dim must divide evenly across the banks in one V channel group"
                )
        if self.dram_columns % self.burst_length:
            raise ValueError("DRAM row length must be a multiple of the burst length")
        if self.dram_columns < self.shape.head_dim * self.k_contexts_per_row:
            raise ValueError("one K row cannot hold the requested packed contexts")

    @property
    def total_banks(self) -> int:
        return self.num_channels * self.banks_per_channel

    @property
    def banks_per_kv_head(self) -> int:
        return self.total_banks // self.shape.local_kv_heads

    @property
    def channels_per_kv_head(self) -> int:
        return self.num_channels // self.shape.local_kv_heads

    @property
    def q_channels_per_gqa_group(self) -> int:
        return self.channels_per_kv_head

    @property
    def q_columns_per_bank(self) -> int:
        return math.ceil(
            (self.shape.gqa_factor * self.shape.head_dim) / self.banks_per_kv_head
        )

    @property
    def kv_projection_channels_per_operand(self) -> int:
        """Active full-bank channels in each half-device K/V projection."""

        half = max(1, self.num_channels // 2)
        return min(
            half,
            math.ceil(self.shape.local_kv_dim / self.banks_per_channel),
        )

    @property
    def kv_projection_columns_per_bank(self) -> int:
        active_banks = self.kv_projection_channels_per_operand * self.banks_per_channel
        return math.ceil(self.shape.local_kv_dim / active_banks)

    @property
    def k_rows_per_bank(self) -> int:
        contexts_per_wave = self.banks_per_kv_head * self.k_contexts_per_row
        return math.ceil(self.max_seq_len / contexts_per_wave)

    @property
    def v_context_split(self) -> bool:
        return self.banks_per_kv_head > self.shape.head_dim

    @property
    def v_context_shards(self) -> int:
        if not self.v_context_split:
            return 1
        return self.banks_per_kv_head // self.shape.head_dim

    @property
    def v_channels_per_shard(self) -> int:
        if not self.v_context_split:
            return self.channels_per_kv_head
        return self.shape.head_dim // self.banks_per_channel

    @property
    def v_dimensions_per_channel(self) -> int:
        if self.v_context_split:
            return self.banks_per_channel
        return math.ceil(self.shape.head_dim / self.channels_per_kv_head)

    @property
    def v_dimension_iterations(self) -> int:
        if self.v_context_split:
            return 1
        return math.ceil(self.shape.head_dim / self.banks_per_kv_head)

    @property
    def v_context_capacity_per_bank(self) -> int:
        if not self.v_context_split:
            return self.max_seq_len
        return math.ceil(self.max_seq_len / self.v_context_shards)

    @property
    def v_dimensions_per_row(self) -> int:
        """Packed V dimension shards per row for short contexts."""

        return 1

    @property
    def v_rows_per_bank(self) -> int:
        if not self.v_context_split:
            rows_per_dimension = math.ceil(self.max_seq_len / self.dram_columns)
            return rows_per_dimension * self.v_dimension_iterations
        return math.ceil(self.v_context_capacity_per_bank / self.dram_columns)

    @property
    def v_reduction_width(self) -> int:
        return self.v_context_shards

    @property
    def v_reduction_adds_per_rank(self) -> int:
        return self.v_reduction_adds(self.max_seq_len)

    def v_reduction_adds(self, sequence_length: int) -> int:
        """Vector additions needed for the active V context on one rank."""

        if not 0 < sequence_length <= self.max_seq_len:
            raise ValueError(
                f"active sequence length must be in [1, {self.max_seq_len}]"
            )
        if not self.v_context_split:
            return 0
        active_banks = min(
            self.v_context_shards,
            math.ceil(sequence_length / self.v_context_capacity_per_bank),
        )
        return (
            self.shape.local_query_heads
            * self.shape.head_dim
            * (active_banks - 1)
        )

    def kv_head_channels(self, local_kv_head: int) -> tuple[int, ...]:
        if not 0 <= local_kv_head < self.shape.local_kv_heads:
            raise ValueError(f"local KV head is out of range: {local_kv_head}")
        first = local_kv_head * self.channels_per_kv_head
        return tuple(range(first, first + self.channels_per_kv_head))

    def kv_head_bank_base(self, local_kv_head: int) -> int:
        if not 0 <= local_kv_head < self.shape.local_kv_heads:
            raise ValueError(f"local KV head is out of range: {local_kv_head}")
        return local_kv_head * self.banks_per_kv_head

    def v_shard_channels(
        self, local_kv_head: int, context_shard: int
    ) -> tuple[int, ...]:
        if not self.v_context_split:
            if context_shard != 0:
                raise ValueError("an unsharded V head has only context shard 0")
            return self.kv_head_channels(local_kv_head)
        if not 0 <= context_shard < self.v_context_shards:
            raise ValueError(f"V context shard is out of range: {context_shard}")
        first = (
            local_kv_head * self.channels_per_kv_head
            + context_shard * self.v_channels_per_shard
        )
        return tuple(range(first, first + self.v_channels_per_shard))

    def k_location(self, local_kv_head: int, sequence: int) -> tuple[int, int, int, int]:
        """Return channel, bank, row-within-cache, and packed segment."""

        if not 0 <= sequence < self.max_seq_len:
            raise ValueError(f"sequence index is out of range: {sequence}")
        wave_span = self.banks_per_kv_head * self.k_contexts_per_row
        wave = sequence // wave_span
        within_wave = sequence % wave_span
        segment = within_wave // self.banks_per_kv_head
        relative_bank = within_wave % self.banks_per_kv_head
        global_bank = self.kv_head_bank_base(local_kv_head) + relative_bank
        channel = global_bank // self.banks_per_channel
        bank = global_bank % self.banks_per_channel
        return channel, bank, wave, segment

    def v_location(self, local_kv_head: int, dimension: int, sequence: int) -> tuple[int, int, int, int]:
        """Return channel, bank, row-within-cache, and row-column segment."""

        if not 0 <= dimension < self.shape.head_dim:
            raise ValueError(f"head dimension is out of range: {dimension}")
        if not 0 <= sequence < self.max_seq_len:
            raise ValueError(f"sequence index is out of range: {sequence}")
        if not self.v_context_split:
            relative_bank = dimension % self.banks_per_kv_head
            dim_iteration = dimension // self.banks_per_kv_head
            global_bank = self.kv_head_bank_base(local_kv_head) + relative_bank
            channel = global_bank // self.banks_per_channel
            bank = global_bank % self.banks_per_channel
            rows_per_dimension = math.ceil(self.max_seq_len / self.dram_columns)
            row = dim_iteration * rows_per_dimension + sequence // self.dram_columns
            return channel, bank, row, sequence % self.dram_columns

        shard_capacity = self.v_context_capacity_per_bank
        context_shard = min(sequence // shard_capacity, self.v_context_shards - 1)
        sequence_in_shard = sequence - context_shard * shard_capacity
        channels = self.v_shard_channels(local_kv_head, context_shard)
        channel = channels[dimension // self.banks_per_channel]
        bank = dimension % self.banks_per_channel
        row = sequence_in_shard // self.dram_columns
        column = sequence_in_shard % self.dram_columns
        return channel, bank, row, column


def kv_head_tp_shape(
    *,
    dim: int,
    query_heads: int,
    kv_heads: int,
    ffn_dim: int,
    tp: int,
    head_dim: int = 128,
) -> KVHeadTPShape:
    """Return the local tensor shapes for an even KV-head TP partition."""

    values = {
        "dim": dim,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "ffn_dim": ffn_dim,
        "tp": tp,
        "head_dim": head_dim,
    }
    non_positive = [name for name, value in values.items() if value <= 0]
    if non_positive:
        raise ValueError(f"KV-head TP dimensions must be positive: {', '.join(non_positive)}")
    if dim != query_heads * head_dim:
        raise ValueError(
            f"hidden dimension {dim} must equal query_heads*head_dim "
            f"({query_heads}*{head_dim})"
        )
    if query_heads % kv_heads:
        raise ValueError(
            f"query heads ({query_heads}) must be divisible by KV heads ({kv_heads})"
        )
    for name, value in (
        ("query heads", query_heads),
        ("KV heads", kv_heads),
        ("FFN dimension", ffn_dim),
    ):
        if value % tp:
            raise ValueError(f"{name} ({value}) must be divisible by TP={tp}")

    local_query_heads = query_heads // tp
    local_kv_heads = kv_heads // tp
    return KVHeadTPShape(
        tp=tp,
        dim=dim,
        head_dim=head_dim,
        global_query_heads=query_heads,
        global_kv_heads=kv_heads,
        global_ffn_dim=ffn_dim,
        local_query_heads=local_query_heads,
        local_kv_heads=local_kv_heads,
        local_query_dim=local_query_heads * head_dim,
        local_kv_dim=local_kv_heads * head_dim,
        local_ffn_dim=ffn_dim // tp,
    )


def validate_kv_head_tp_role(tp: int, role: str) -> None:
    if role not in KV_HEAD_TP_ROLES:
        raise ValueError(
            f"Unknown KV-head TP role '{role}'; expected one of: "
            + ", ".join(KV_HEAD_TP_ROLES)
        )
    if tp == 1 and role != "main":
        raise ValueError("TP=1 has only a main trace")
