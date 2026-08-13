import argparse
import unittest
from types import SimpleNamespace

import cxl_latency
import cent_power_calculator
import run_sim
from TransformerBlock import TransformerBlock
from Llama import TransformerBlockLlama
from scripts import run_cent_memory_cases as runner
from scripts import analyze_kv_head_tp_systolic_all_context as combined_analysis
from tp_mapping import (
    KVHeadTPLayout,
    SystolicTPLayout,
    kv_head_tp_shape,
    validate_kv_head_tp_role,
)


class TraceRecorder:
    def __init__(self, layout, banks_per_channel):
        self.kv_head_layout = layout
        self.head_dim = layout.shape.head_dim
        self.burst_length = layout.burst_length
        self.num_banks = banks_per_channel
        self.DRAM_column = layout.dram_columns
        self.max_seq_len = layout.max_seq_len
        self.events = []

    def WR_GB_only_trace(self, channels, op_size):
        self.events.append(("WR_GB", tuple(channels), op_size))

    def WR_BIAS_only_trace(self, channels):
        self.events.append(("WR_BIAS", tuple(channels)))

    def MAC_ABK_only_trace(self, channels, row, op_size, timing):
        self.events.append(("MAC_ABK", tuple(channels), row, op_size, timing))

    def RD_MAC_only_trace(self, channels):
        self.events.append(("RD_MAC", tuple(channels)))


class SystolicTraceRecorder:
    _record_systolic_stage = TransformerBlock._record_systolic_stage
    _MAC_ABK_systolic_stage_only_trace = (
        TransformerBlock._MAC_ABK_systolic_stage_only_trace
    )
    _trace_systolic_pipeline_only_trace = (
        TransformerBlock._trace_systolic_pipeline_only_trace
    )
    Vector_Matrix_Mul_weight_systolic_tp_only_trace = (
        TransformerBlock.Vector_Matrix_Mul_weight_systolic_tp_only_trace
    )
    _trace_fused_activation_systolic_pim_only_trace = (
        TransformerBlock._trace_fused_activation_systolic_pim_only_trace
    )
    Vector_Matrix_Mul_score_systolic_tp_only_trace = (
        TransformerBlock.Vector_Matrix_Mul_score_systolic_tp_only_trace
    )
    Vector_Matrix_Mul_output_systolic_tp_only_trace = (
        TransformerBlock.Vector_Matrix_Mul_output_systolic_tp_only_trace
    )
    Vector_Matrix_Mul_flash_attention_systolic_tp_only_trace = (
        TransformerBlock.Vector_Matrix_Mul_flash_attention_systolic_tp_only_trace
    )
    systolic_tp_cache_rows_per_batch = (
        TransformerBlock.systolic_tp_cache_rows_per_batch
    )
    trace_systolic_tp_cache_update = (
        TransformerBlock.trace_systolic_tp_cache_update
    )

    def __init__(self, layout, batch_size=None):
        self.systolic_tp_layout = layout
        self.kv_head_layout = KVHeadTPLayout(
            shape=layout.shape,
            num_channels=layout.num_channels,
            banks_per_channel=layout.banks_per_channel,
            max_seq_len=layout.max_seq_len,
        )
        self.systolic_dim = layout.systolic_height
        self.batch_size = layout.systolic_height if batch_size is None else batch_size
        self.max_seq_len = layout.max_seq_len
        self.num_banks = layout.banks_per_channel
        self.head_dim = layout.shape.head_dim
        self.DRAM_column = layout.dram_columns
        self.burst_length = layout.burst_length
        self.systolic_pipeline_cycles = {"fill": 0, "reduction": 0, "drain": 0}
        self.events = []

    def WR_GB_only_trace(self, channels, op_size):
        self.events.append(("WR_GB", tuple(channels), op_size))

    def WR_BIAS_only_trace(self, channels):
        self.events.append(("WR_BIAS", tuple(channels)))

    def MAC_ABK_only_trace(self, channels, row, op_size, timing):
        self.events.append(("MAC_ABK", tuple(channels), row, op_size, timing))

    def RD_MAC_only_trace(self, channels):
        self.events.append(("RD_MAC", tuple(channels)))

    def AF_only_trace(self, channels):
        self.events.append(("AF", tuple(channels)))

    def RD_AF_only_trace(self, channels):
        self.events.append(("RD_AF", tuple(channels)))

    def W_MEM_only_trace(self, channel, bank, row, elements):
        self.events.append(("W_MEM", channel, bank, row, elements))


def channels_before_first_drain(events):
    first_drain = next(index for index, event in enumerate(events) if event[0] == "RD_MAC")
    mac_masks = [event[1] for event in events[:first_drain] if event[0] == "MAC_ABK"]
    return mac_masks, {channel for mask in mac_masks for channel in mask}


class KVHeadTPShapeTests(unittest.TestCase):
    def test_70b_tp8_has_one_local_kv_head(self):
        shape = kv_head_tp_shape(
            dim=8192,
            query_heads=64,
            kv_heads=8,
            ffn_dim=28672,
            tp=8,
        )
        self.assertEqual(shape.local_query_heads, 8)
        self.assertEqual(shape.local_kv_heads, 1)
        self.assertEqual(shape.local_query_dim, 1024)
        self.assertEqual(shape.local_kv_dim, 128)
        self.assertEqual(shape.local_ffn_dim, 3584)
        self.assertEqual(shape.gqa_factor, 8)

    def test_ragged_kv_head_partition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            kv_head_tp_shape(
                dim=8192,
                query_heads=64,
                kv_heads=8,
                ffn_dim=28672,
                tp=3,
            )

    def test_tp1_has_no_helper(self):
        with self.assertRaisesRegex(ValueError, "TP=1"):
            validate_kv_head_tp_role(1, "helper")

    def test_70b_gddr6_v_layout_switches_to_context_shards_at_tp4(self):
        layouts = {}
        for tp in (1, 2, 4, 8):
            layouts[tp] = KVHeadTPLayout(
                shape=kv_head_tp_shape(
                    dim=8192,
                    query_heads=64,
                    kv_heads=8,
                    ffn_dim=28672,
                    tp=tp,
                ),
                num_channels=32,
                banks_per_channel=16,
                max_seq_len=131072,
            )
        self.assertEqual(
            [layouts[tp].banks_per_kv_head for tp in (1, 2, 4, 8)],
            [64, 128, 256, 512],
        )
        self.assertEqual(
            [layouts[tp].v_context_split for tp in (1, 2, 4, 8)],
            [False, False, True, True],
        )
        self.assertEqual(layouts[4].v_reduction_width, 2)
        self.assertEqual(layouts[8].v_reduction_width, 4)
        self.assertEqual(layouts[8].v_channels_per_shard, 8)

    def test_lpddr4x_tp8_uses_two_128_bank_channel_groups(self):
        layout = KVHeadTPLayout(
            shape=kv_head_tp_shape(
                dim=8192,
                query_heads=64,
                kv_heads=8,
                ffn_dim=28672,
                tp=8,
            ),
            num_channels=32,
            banks_per_channel=8,
            max_seq_len=32768,
        )
        self.assertTrue(layout.v_context_split)
        self.assertEqual(layout.v_reduction_width, 2)
        self.assertEqual(layout.v_channels_per_shard, 16)
        self.assertEqual(layout.v_dimensions_per_channel, 8)

    def test_short_context_uses_one_row_per_channel_group_bank(self):
        layout = KVHeadTPLayout(
            shape=kv_head_tp_shape(
                dim=8192,
                query_heads=64,
                kv_heads=8,
                ffn_dim=28672,
                tp=8,
            ),
            num_channels=32,
            banks_per_channel=16,
            max_seq_len=4096,
        )
        self.assertEqual(layout.v_context_capacity_per_bank, 1024)
        self.assertEqual(layout.v_dimensions_per_row, 1)
        self.assertEqual(layout.v_rows_per_bank, 1)
        self.assertEqual(layout.v_location(0, 0, 0), (0, 0, 0, 0))
        self.assertEqual(layout.v_location(0, 127, 0), (7, 15, 0, 0))
        self.assertEqual(layout.v_location(0, 0, 1024), (8, 0, 0, 0))
        self.assertEqual(layout.v_location(0, 127, 1024), (15, 15, 0, 0))


class SystolicTPLayoutTests(unittest.TestCase):
    @staticmethod
    def layout(tp, systolic_height=4, max_seq_len=4096, banks_per_channel=16):
        return SystolicTPLayout(
            shape=kv_head_tp_shape(
                dim=8192,
                query_heads=64,
                kv_heads=8,
                ffn_dim=28672,
                tp=tp,
            ),
            num_channels=32,
            banks_per_channel=banks_per_channel,
            max_seq_len=max_seq_len,
            systolic_height=systolic_height,
        )

    def test_tp8_projection_groups_match_standard_mapping(self):
        layout = self.layout(8)
        self.assertEqual(
            (layout.q.channels_per_group, layout.q.reduction_groups, layout.q.reduction_slice),
            (4, 8, 1024),
        )
        self.assertEqual(
            (layout.kv.channels_per_group, layout.kv.reduction_groups, layout.kv.reduction_slice),
            (1, 32, 256),
        )
        self.assertEqual(layout.wo.input_dim, 1024)
        self.assertEqual(layout.wo.output_dim, 8192)
        self.assertEqual(layout.w2.input_dim, 3584)

    def test_fused_w1_w3_uses_one_tp8_tile_and_seven_tp1_tiles(self):
        tp8 = self.layout(8)
        self.assertEqual(tp8.fused_ffn.output_dim, 7168)
        self.assertEqual(tp8.fused_ffn.output_tiles, 1)
        self.assertEqual(tp8.fused_ffn.active_channels, 28)
        self.assertEqual(tp8.fused_ffn.pnm_reduction_adds, 0)

        tp1 = self.layout(1)
        self.assertEqual(tp1.fused_ffn.output_dim, 57344)
        self.assertEqual(tp1.fused_ffn.output_tiles, 7)
        self.assertEqual(tp1.fused_ffn.active_channels, 32)

    def test_sa4_tp8_packs_query_groups_into_both_halves(self):
        layout = self.layout(8, systolic_height=4)
        sv = layout.sv(4096)
        self.assertEqual(sv.mode, "paired_query_groups")
        self.assertEqual(sv.context_groups_per_kv_head, 32)
        self.assertEqual(sv.contexts_per_half, 128)
        self.assertEqual(sv.query_waves, 1)
        self.assertEqual(sv.height_utilization, 1.0)
        self.assertEqual(sv.width_utilization, 1.0)
        self.assertEqual(layout.sv_reduction_adds(4096), 31 * 1024)

    def test_sa8_tp8_packs_two_context_shards_per_channel(self):
        layout = self.layout(8, systolic_height=8)
        sv = layout.sv(4096)
        self.assertEqual(sv.mode, "paired_context_shards")
        self.assertEqual(sv.context_groups_per_kv_head, 64)
        self.assertEqual(sv.contexts_per_half, 64)
        self.assertEqual(sv.query_waves, 1)
        self.assertEqual(sv.height_utilization, 1.0)
        self.assertEqual(sv.width_utilization, 1.0)
        self.assertEqual(layout.sv_reduction_adds(4096), 63 * 1024)

    def test_tp1_pairs_local_kv_heads(self):
        layout = self.layout(1, systolic_height=4)
        sv = layout.sv(4096)
        self.assertEqual(sv.mode, "paired_kv_heads")
        self.assertEqual(sv.channels_per_work_item, 8)
        self.assertEqual(sv.context_groups_per_kv_head, 8)
        self.assertEqual(sv.contexts_per_half, 512)

    def test_tp8_qk_width_is_half_at_4k(self):
        tp8 = self.layout(8)
        self.assertEqual(tp8.k_contexts_per_bank(4096), 8)
        self.assertEqual(tp8.k_width_utilization(4096), 0.5)
        self.assertEqual(self.layout(4).k_width_utilization(4096), 1.0)

    def test_lpddr4x_uses_two_channel_sv_groups_and_output_tiles(self):
        layout = self.layout(8, banks_per_channel=8)
        self.assertEqual(layout.channels_per_sv_group, 2)
        self.assertEqual(layout.wo.output_tiles, 2)
        self.assertEqual(layout.w2.output_tiles, 2)
        self.assertEqual(layout.fused_ffn.output_tiles, 2)
        self.assertEqual(layout.fused_ffn.active_channels, 32)
        sv = layout.sv(4096)
        self.assertEqual(sv.mode, "paired_query_groups")
        self.assertEqual(sv.channels_per_context_group, 2)
        self.assertEqual(sv.context_groups_per_kv_head, 16)
        self.assertEqual(sv.contexts_per_half, 256)
        self.assertEqual(sv.width_utilization, 1.0)

    def test_lpddr4x_tp1_pairs_heads_over_four_groups_each(self):
        sv = self.layout(1, banks_per_channel=8).sv(4096)
        self.assertEqual(sv.mode, "paired_kv_heads")
        self.assertEqual(sv.channels_per_work_item, 8)
        self.assertEqual(sv.channels_per_context_group, 2)
        self.assertEqual(sv.context_groups_per_kv_head, 4)
        self.assertEqual(sv.contexts_per_half, 1024)

    def test_projection_and_sv_pnm_work_is_explicit(self):
        work = self.layout(8, systolic_height=4).pnm_reduction_adds(4096)
        self.assertEqual(work["q"], 7 * 1024)
        self.assertEqual(work["kv"], 31 * 256)
        self.assertEqual(work["sv"], 31 * 1024)
        self.assertEqual(work["fused_ffn"], 0)


class SystolicTPTraceTests(unittest.TestCase):
    @staticmethod
    def layout(tp=8, systolic_height=4, banks_per_channel=16):
        return SystolicTPLayoutTests.layout(
            tp, systolic_height, banks_per_channel=banks_per_channel
        )

    def test_q_projection_emits_fill_body_and_drain_per_gb_chunk(self):
        recorder = SystolicTraceRecorder(self.layout(), batch_size=4)
        recorder.Vector_Matrix_Mul_weight_systolic_tp_only_trace(
            recorder.systolic_tp_layout.q, 100, "weight"
        )
        self.assertEqual(
            recorder.systolic_pipeline_cycles,
            {"fill": 12, "reduction": 1024, "drain": 12},
        )
        self.assertEqual(
            len([event for event in recorder.events if event[0] == "WR_GB"]), 4
        )
        mac_masks = {
            event[1] for event in recorder.events if event[0] == "MAC_ABK"
        }
        self.assertEqual(mac_masks, {tuple(range(32))})

    def test_projection_batch_rows_change_chunks_without_repeating_weights(self):
        expected = {
            1: ({"fill": 0, "reduction": 1024, "drain": 0}, 1),
            2: ({"fill": 2, "reduction": 1024, "drain": 2}, 2),
            3: ({"fill": 8, "reduction": 1024, "drain": 8}, 4),
            4: ({"fill": 12, "reduction": 1024, "drain": 12}, 4),
        }
        for batch_size, (cycles, gb_commands) in expected.items():
            with self.subTest(batch_size=batch_size):
                recorder = SystolicTraceRecorder(
                    self.layout(), batch_size=batch_size
                )
                recorder.Vector_Matrix_Mul_weight_systolic_tp_only_trace(
                    recorder.systolic_tp_layout.q, 100, "weight"
                )
                self.assertEqual(recorder.systolic_pipeline_cycles, cycles)
                self.assertEqual(
                    len(
                        [
                            event
                            for event in recorder.events
                            if event[0] == "WR_GB"
                        ]
                    ),
                    gb_commands,
                )
                self.assertEqual(
                    recorder.systolic_kernel_active_rows["q"], [batch_size]
                )

    def test_cache_batches_use_contiguous_noninterleaved_row_blocks(self):
        recorder = SystolicTraceRecorder(self.layout(), batch_size=2)
        k_stride, v_stride = recorder.systolic_tp_cache_rows_per_batch()

        recorder.trace_systolic_tp_cache_update(1000, 2000, 4095, 0)
        first = list(recorder.events)
        recorder.events.clear()
        recorder.trace_systolic_tp_cache_update(1000, 2000, 4095, 1)
        second = list(recorder.events)

        self.assertEqual(first[0][0], "W_MEM")
        self.assertEqual(second[0][3] - first[0][3], k_stride)
        first_v = next(event for event in first[1:] if event[0] == "W_MEM")
        second_v = next(event for event in second[1:] if event[0] == "W_MEM")
        self.assertEqual(second_v[3] - first_v[3], v_stride)

    def test_attention_serializes_batches_over_contiguous_cache_blocks(self):
        class ScheduleRecorder:
            trace_only_systolic_TP = TransformerBlockLlama.trace_only_systolic_TP

            def __init__(self):
                self.systolic_tp_layout = SimpleNamespace(
                    q="q", kv="kv", wo="wo", fused_ffn="ffn", w2="w2"
                )
                self.seqlen = 64
                self.batch_size = 4
                self.trace_fc_kqvo = True
                self.trace_attention = True
                self.flash_attention = False
                self.trace_score = False
                self.trace_norm = False
                self.trace_fc_ffn = False
                self.wq_row_index = 100
                self.wk_row_index = 200
                self.cache_k_row_index = 1000
                self.cache_v_row_index = 2000
                self.scores_row_index = 3000
                self.wo_row_index = 4000
                self.FC_total_banks = 512
                self.events = []

            def Vector_Matrix_Mul_weight_systolic_tp_only_trace(
                self, projection, row, timing
            ):
                self.events.append(("projection", projection, row))

            def trace_systolic_tp_cache_update(self, k, v, sequence, batch):
                self.events.append(("cache", batch, k, v, sequence))

            def systolic_tp_cache_rows_per_batch(self):
                return 10, 20

            def Vector_Matrix_Mul_score_systolic_tp_only_trace(self, row, seqlen):
                self.events.append(("qk", row, seqlen))

            def store_for_score_only_trace(self, row, banks, seqlen):
                self.events.append(("score_store", row, seqlen))

            def load_for_score_only_trace(self, row, banks, seqlen):
                self.events.append(("score_load", row, seqlen))

            def SYNC_only_trace(self):
                self.events.append(("sync",))

            def Vector_Matrix_Mul_output_systolic_tp_only_trace(self, row, seqlen):
                self.events.append(("sv", row, seqlen))

        recorder = ScheduleRecorder()
        recorder.trace_only_systolic_TP()
        self.assertEqual(
            [event[1] for event in recorder.events if event[0] == "cache"],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            [event[1] for event in recorder.events if event[0] == "qk"],
            [1000, 1010, 1020, 1030],
        )
        self.assertEqual(
            [event[1] for event in recorder.events if event[0] == "sv"],
            [2000, 2020, 2040, 2060],
        )

    def test_flash_attention_directly_connects_qk_to_sv_per_block(self):
        class FlashScheduleRecorder:
            Vector_Matrix_Mul_flash_attention_systolic_tp_only_trace = (
                TransformerBlock.Vector_Matrix_Mul_flash_attention_systolic_tp_only_trace
            )

            def __init__(self, layout):
                self.systolic_tp_layout = layout
                self.flash_attention_block_size = 1024
                self.trace_score = False
                self.events = []

            def Vector_Matrix_Mul_score_systolic_tp_only_trace(self, row, seqlen):
                self.events.append(("qk", row, seqlen))

            def Vector_Matrix_Mul_output_systolic_tp_only_trace(self, row, seqlen):
                self.events.append(("sv", row, seqlen))

        # TP=1 has eight local KV heads.  The cent_dev-compatible K stride is
        # ceil(1024 / 512 banks) * ceil(128 * 8 / 1024) = 2 rows; V advances
        # by ceil(1024 / 1024) = 1 command-address row per flash block.
        recorder = FlashScheduleRecorder(self.layout(tp=1))
        recorder.Vector_Matrix_Mul_flash_attention_systolic_tp_only_trace(
            100, 200, 4096
        )
        self.assertEqual(
            recorder.events,
            [
                ("qk", 100, 1024), ("sv", 200, 1024),
                ("qk", 102, 1024), ("sv", 201, 1024),
                ("qk", 104, 1024), ("sv", 202, 1024),
                ("qk", 106, 1024), ("sv", 203, 1024),
            ],
        )

    def test_flash_attention_score_trace_omits_sv(self):
        class ScoreOnlyRecorder:
            Vector_Matrix_Mul_flash_attention_systolic_tp_only_trace = (
                TransformerBlock.Vector_Matrix_Mul_flash_attention_systolic_tp_only_trace
            )

            def __init__(self, layout):
                self.systolic_tp_layout = layout
                self.flash_attention_block_size = 1024
                self.trace_score = True
                self.events = []

            def Vector_Matrix_Mul_score_systolic_tp_only_trace(self, row, seqlen):
                self.events.append(("qk", row, seqlen))

            def Vector_Matrix_Mul_output_systolic_tp_only_trace(self, row, seqlen):
                self.events.append(("sv", row, seqlen))

        recorder = ScoreOnlyRecorder(self.layout(tp=1))
        recorder.Vector_Matrix_Mul_flash_attention_systolic_tp_only_trace(
            100, 200, 2048
        )
        self.assertEqual(recorder.events, [("qk", 100, 1024), ("qk", 102, 1024)])

    def test_fused_tp8_ffn_uses_28_channels_and_one_output_tile(self):
        recorder = SystolicTraceRecorder(self.layout())
        recorder.Vector_Matrix_Mul_weight_systolic_tp_only_trace(
            recorder.systolic_tp_layout.fused_ffn, 200, "ffn"
        )
        masks = {
            event[1]
            for event in recorder.events
            if event[0] in ("WR_GB", "MAC_ABK")
        }
        self.assertEqual(masks, {tuple(range(28))})
        self.assertEqual(recorder.systolic_pipeline_cycles["reduction"], 8192)

    def test_w1_silu_af_rd_af_remains_with_ewmul_pnm(self):
        recorder = SystolicTraceRecorder(self.layout())
        recorder._trace_fused_activation_systolic_pim_only_trace(
            recorder.systolic_tp_layout.shape.local_ffn_dim
        )
        af = [event for event in recorder.events if event[0] == "AF"]
        rd_af = [event for event in recorder.events if event[0] == "RD_AF"]
        # TP=8 has a 3,584-column W1 shard: one 14-channel output tile,
        # with four AF/RD_AF commands for the 4x16 array rows.
        self.assertEqual(af, [("AF", tuple(range(14)))] * 4)
        self.assertEqual(rd_af, [("RD_AF", tuple(range(14)))] * 4)

    def test_sa4_sv_query_pair_is_one_full_width_wave(self):
        recorder = SystolicTraceRecorder(self.layout(systolic_height=4))
        recorder.Vector_Matrix_Mul_output_systolic_tp_only_trace(300, 4096)
        self.assertEqual(
            recorder.systolic_pipeline_cycles,
            {"fill": 3, "reduction": 128, "drain": 3},
        )
        gb = [event for event in recorder.events if event[0] == "WR_GB"]
        self.assertEqual(gb, [("WR_GB", tuple(range(32)), 64)])

    def test_sa8_sv_context_pair_fits_one_gb_and_one_wave(self):
        recorder = SystolicTraceRecorder(self.layout(systolic_height=8))
        recorder.Vector_Matrix_Mul_output_systolic_tp_only_trace(300, 4096)
        self.assertEqual(
            recorder.systolic_pipeline_cycles,
            {"fill": 7, "reduction": 64, "drain": 7},
        )
        gb = [event for event in recorder.events if event[0] == "WR_GB"]
        self.assertEqual(gb, [("WR_GB", tuple(range(32)), 64)])

    def test_tp8_qk_reports_two_query_waves_and_half_width_layout(self):
        recorder = SystolicTraceRecorder(self.layout(systolic_height=4))
        recorder.Vector_Matrix_Mul_score_systolic_tp_only_trace(400, 4096)
        self.assertEqual(
            recorder.systolic_pipeline_cycles,
            {"fill": 6, "reduction": 256, "drain": 6},
        )

    def test_lpddr4x_sv_uses_sixteen_two_channel_groups(self):
        recorder = SystolicTraceRecorder(
            self.layout(systolic_height=4, banks_per_channel=8)
        )
        recorder.Vector_Matrix_Mul_output_systolic_tp_only_trace(300, 4096)
        self.assertEqual(
            recorder.systolic_pipeline_cycles,
            {"fill": 6, "reduction": 256, "drain": 6},
        )

    def test_lpddr4x_row_parallel_projections_use_two_tiles(self):
        recorder = SystolicTraceRecorder(
            self.layout(systolic_height=4, banks_per_channel=8)
        )
        recorder.Vector_Matrix_Mul_weight_systolic_tp_only_trace(
            recorder.systolic_tp_layout.wo, 500, "wo"
        )
        gb = [event for event in recorder.events if event[0] == "WR_GB"]
        self.assertEqual(len(gb), 8)
        self.assertEqual({event[1] for event in gb}, {tuple(range(32))})


class KVHeadTPTraceSchedulingTests(unittest.TestCase):
    @staticmethod
    def layout(tp, banks_per_channel):
        return KVHeadTPLayout(
            shape=kv_head_tp_shape(
                dim=8192,
                query_heads=64,
                kv_heads=8,
                ffn_dim=28672,
                tp=tp,
            ),
            num_channels=32,
            banks_per_channel=banks_per_channel,
            max_seq_len=4096,
        )

    def test_qk_coissues_all_local_heads_before_first_drain(self):
        for banks_per_channel in (8, 16):
            for tp in (1, 2, 4, 8):
                with self.subTest(banks_per_channel=banks_per_channel, tp=tp):
                    recorder = TraceRecorder(
                        self.layout(tp, banks_per_channel), banks_per_channel
                    )
                    TransformerBlock._Vector_Matrix_Mul_score_kv_head_tp_only_trace(
                        recorder, 100, 512, "attention"
                    )
                    mac_masks, active_channels = channels_before_first_drain(
                        recorder.events
                    )
                    self.assertEqual(len(mac_masks), 1)
                    self.assertEqual(active_channels, set(range(32)))
                    self.assertEqual(set(mac_masks[0]), set(range(32)))
                    first_drain = next(
                        event for event in recorder.events if event[0] == "RD_MAC"
                    )
                    self.assertEqual(set(first_drain[1]), set(range(32)))

    def test_sv_coissues_all_heads_and_context_groups_before_first_drain(self):
        for banks_per_channel in (8, 16):
            for tp in (1, 2, 4, 8):
                with self.subTest(banks_per_channel=banks_per_channel, tp=tp):
                    recorder = TraceRecorder(
                        self.layout(tp, banks_per_channel), banks_per_channel
                    )
                    TransformerBlock._Vector_Matrix_Mul_output_kv_head_tp_only_trace(
                        recorder, 200, 4096, "attention"
                    )
                    mac_masks, active_channels = channels_before_first_drain(
                        recorder.events
                    )
                    self.assertEqual(active_channels, set(range(32)))
                    self.assertGreaterEqual(len(mac_masks), 1)
                    first_drain = next(
                        event for event in recorder.events if event[0] == "RD_MAC"
                    )
                    self.assertEqual(set(first_drain[1]), set(range(32)))


class KVHeadTPCXLTests(unittest.TestCase):
    def test_system_payload_is_four_collective_vectors(self):
        main, helper, system = cxl_latency.kv_head_tp_pcie_bits(8192, 4)
        self.assertEqual(main, 2 * 3 * 8192 * 16)
        self.assertEqual(helper, 2 * 8192 * 16)
        self.assertEqual(system, 4 * 3 * 8192 * 16)
        self.assertEqual(system, main + 3 * helper)

    def test_tp1_has_no_cxl_collective(self):
        self.assertEqual(cxl_latency.kv_head_tp_pcie_bits(4096, 1), (0, 0, 0))
        self.assertEqual(cxl_latency.kv_head_tp_latency(4096, 18, 1, 16), 0.0)

    def test_batch_payload_scales_only_the_two_hidden_vector_collectives(self):
        batch = 4
        dim = 8192
        tp = 4
        _main, _helper, system = cxl_latency.kv_head_tp_pcie_bits(
            batch * dim, tp
        )
        self.assertEqual(system, 4 * (tp - 1) * batch * dim * 16)


class SystolicTPPNMEnergyTests(unittest.TestCase):
    def test_cent_dev_shared_buffer_exposes_all_32_channels(self):
        cent_power_calculator.set_channel_count(32, 32)
        self.assertEqual(
            cent_power_calculator.SHARED_BUFFER_CAPACITY_BYTES,
            4 * 1024 * 1024,
        )
        self.assertEqual(run_sim.ACCEL_CYCLE["EXP"], 13.0)
        self.assertEqual(run_sim.ACCEL_CYCLE["VEC"], 4.0)

        args = SimpleNamespace(
            model="Llama2-70B",
            num_channels=32,
            num_banks=16,
            max_seq_len=4096,
            kv_head_tp=True,
            systolic_pim=True,
            systolic_dim=4,
            ewmul_pnm=False,
            flash_attention=False,
        )
        latency = run_sim.calculate_acc_latency(args, 4096, tp=1)
        # EXP + VEC_ADD + two VEC_MUL normalization operations, followed by
        # the per-head reduction and RISC-V completion work.
        self.assertAlmostEqual(latency["Softmax_latency"], 0.006984)

    def test_pipelined_softmax_uses_cent_dev_systolic_startup_window(self):
        common = {
            "model": "Llama2-70B",
            "num_channels": 32,
            "max_seq_len": 131072,
            "kv_head_tp": True,
            "systolic_pim": True,
            "systolic_dim": 4,
            "ewmul_pnm": False,
            "flash_attention": True,
            "flash_attention_block_size": 1024,
        }
        expected_factors = {
            # GDDR6: 16 burst elements x 32 channels x 16 banks = 8192.
            16: {4096: 1.0, 32768: 1.0 / 4.0, 131072: 1.0 / 16.0},
            # LPDDR4X: 16 x 32 x 8 = 4096.
            8: {4096: 1.0, 32768: 1.0 / 8.0, 131072: 1.0 / 32.0},
        }
        for banks_per_channel, factors in expected_factors.items():
            for seqlen, expected_factor in factors.items():
                with self.subTest(banks=banks_per_channel, seqlen=seqlen):
                    base = run_sim.calculate_acc_latency(
                        SimpleNamespace(
                            num_banks=banks_per_channel,
                            pipelined_softmax=False,
                            **common,
                        ),
                        seqlen,
                        tp=1,
                    )
                    pipelined = run_sim.calculate_acc_latency(
                        SimpleNamespace(
                            num_banks=banks_per_channel,
                            pipelined_softmax=True,
                            **common,
                        ),
                        seqlen,
                        tp=1,
                    )
                    self.assertAlmostEqual(
                        pipelined["Softmax_latency"],
                        base["Softmax_latency"] * expected_factor,
                    )
                    # Only Softmax is overlapped; PNM reductions and block
                    # merge work remain unchanged.
                    self.assertAlmostEqual(
                        pipelined["SVReduction_latency"],
                        base["SVReduction_latency"],
                    )
                    self.assertAlmostEqual(
                        pipelined["FlashAttention_latency"],
                        base["FlashAttention_latency"],
                    )

    def test_flash_block_must_fit_the_four_mib_shared_buffer(self):
        args = SimpleNamespace(
            model="Llama2-70B",
            num_channels=32,
            num_banks=16,
            max_seq_len=65536,
            kv_head_tp=True,
            systolic_pim=True,
            systolic_dim=4,
            ewmul_pnm=False,
            flash_attention=True,
            flash_attention_block_size=65536,
        )
        with self.assertRaisesRegex(ValueError, "shared buffer"):
            run_sim.calculate_acc_latency(args, 65536, tp=1)

    def test_systolic_forces_ewmul_pnm_and_explicit_flag_controls_vector_path(self):
        common = {
            "model": "Llama2-70B",
            "num_channels": 32,
            "num_banks": 16,
            "max_seq_len": 4096,
            "kv_head_tp": True,
            "systolic_dim": 4,
            "flash_attention": False,
        }
        forced = run_sim.calculate_acc_latency(
            SimpleNamespace(
                systolic_pim=True,
                ewmul_pnm=False,
                **common,
            ),
            4096,
            8,
        )
        disabled = run_sim.calculate_acc_latency(
            SimpleNamespace(
                systolic_pim=False,
                ewmul_pnm=False,
                **common,
            ),
            4096,
            8,
        )
        explicit = run_sim.calculate_acc_latency(
            SimpleNamespace(
                systolic_pim=False,
                ewmul_pnm=True,
                **common,
            ),
            4096,
            8,
        )
        self.assertTrue(
            run_sim.ewmul_pnm_enabled(
                SimpleNamespace(systolic_pim=True, ewmul_pnm=False)
            )
        )
        self.assertGreater(forced["EWMULActivation_latency"], 0.0)
        self.assertNotIn("EWMULActivation_latency", disabled)
        self.assertEqual(
            explicit["EWMULActivation_latency"],
            forced["EWMULActivation_latency"],
        )

    def test_flash_attention_uses_the_cent_dev_block_merge_term(self):
        common = {
            "model": "Llama2-70B",
            "num_channels": 32,
            "num_banks": 16,
            "max_seq_len": 4096,
            "kv_head_tp": True,
            "systolic_pim": True,
            "systolic_dim": 4,
            "ewmul_pnm": False,
            "flash_attention": True,
            "flash_attention_block_size": 1024,
        }
        latency = run_sim.calculate_acc_latency(
            SimpleNamespace(**common), 4096, tp=1
        )
        expected = (
            4 * 64 / 16.0 / 32.0 * run_sim.ACCEL_CYCLE["VEC"]
            / (run_sim.FREQ / run_sim.KILO)
        )
        self.assertAlmostEqual(latency["FlashAttention_latency"], expected)

    def test_ewmul_and_reductions_are_charged_analytically(self):
        energy = cent_power_calculator.kv_head_tp_pnm_dynamic_energy(
            {"tCK_ps": 500.0},
            reduction_adds=16,
            ewmul_elements=16,
        )
        self.assertGreater(energy["SB_DYN"], 0.0)
        self.assertGreater(energy["IB_DYN"], 0.0)
        self.assertEqual(energy["EXP_DYN"], 0.0)
        self.assertGreater(energy["VEC_DYN"], 0.0)
        self.assertGreater(energy["VEC_MUL_DYN"], 0.0)


class KVHeadTPRunnerTests(unittest.TestCase):
    @staticmethod
    def args(tp_values):
        return argparse.Namespace(
            num_devices=32,
            tp_values=tp_values,
            inter_device_attention=False,
            kv_head_tp=True,
            model="Llama2-70B",
        )

    def test_modes_are_isolated_from_old_model_parallel_paths(self):
        args = self.args([1, 2, 4, 8])
        self.assertEqual(run_sim.model_parallel_modes_for_tp(args, 1), [run_sim.KV_HEAD_MAIN_MODE])
        self.assertEqual(
            run_sim.model_parallel_modes_for_tp(args, 8),
            [run_sim.KV_HEAD_MAIN_MODE],
        )

    def test_batch_group_energy_is_reported_per_output_token(self):
        group, token = run_sim.normalize_batch_group_energy(
            {"PIM": 12.0, "VEC_DYN": 4.0}, 4
        )
        self.assertEqual(group, 16.0)
        self.assertEqual(token, 4.0)

    def test_tp_above_eight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "<=8"):
            run_sim.model_parallel_tp_values(self.args([16]))

    def test_full_fresh_symmetric_plan_has_72_ramulator_jobs(self):
        models = ["Llama2-7B", "Llama2-70B"]
        contexts = [4096, 32768, 131072]
        values = {(model, context): [1, 2, 4, 8] for model in models for context in contexts}
        self.assertEqual(
            runner.planned_ramulator_jobs(
                models,
                contexts,
                values,
                3,
                include_pipeline=False,
                symmetric_model_parallel=True,
            ),
            72,
        )

    def test_four_batch_sweep_has_288_ramulator_jobs(self):
        models = ["Llama2-7B", "Llama2-70B"]
        contexts = [4096, 32768, 131072]
        values = {
            (model, context): [1, 2, 4, 8]
            for model in models
            for context in contexts
        }
        self.assertEqual(
            runner.planned_ramulator_jobs(
                models,
                contexts,
                values,
                3,
                include_pipeline=False,
                symmetric_model_parallel=True,
                batch_count=4,
            ),
            288,
        )

    def test_combined_analysis_rejects_old_kv_head_vector_source(self):
        with self.assertRaisesRegex(ValueError, "balanced_equal_power_all_contexts"):
            combined_analysis.reject_forbidden_vector_source(
                "output/kv_head_tp_equal_power_all_contexts/analysis/all_candidates.csv"
            )

    def test_batch_groups_can_underfill_an_otherwise_admitted_pipeline(self):
        import pandas as pd
        from pathlib import Path

        source = pd.Series(
            {
                "Main PIM latency": 7.0,
                "Helper PIM latency": 6.0,
                "CXL latency": 2.0,
                "Acc latency": 1.0,
                "Main Acc latency": 1.0,
                "Helper Acc latency": 1.0,
                "TransformerBlock latency": 10.0,
                "Token latency (ms)": 800.0,
                "Token energy (mJ)": 100.0,
                "Pipeline parallelism": 16,
                "Tensor parallelism": 2,
                "Device number": 32,
                "Batch size": 4,
            }
        )
        row = combined_analysis.build_systolic_batch_candidate(
            "GDDR6",
            "GDDR6",
            Path("/tmp/source.csv"),
            source,
            "Llama2-70B",
            "128K",
            16,
            2,
            4,
            16.0,
            0.0,
        )
        self.assertEqual(row["Max resident requests"], 9)
        self.assertEqual(row["Max resident batch groups"], 2)
        self.assertEqual(row["Resident requests used"], 8)
        self.assertAlmostEqual(row["Pipeline fill ratio"], 2 / 16)
        self.assertAlmostEqual(row["Projection SA row utilization"], 1.0)
        self.assertAlmostEqual(
            row["Replica throughput (tokens/s)"],
            row["Ideal full-pipeline throughput (tokens/s)"] * 2 / 16,
        )

    def test_equal_power_selection_scales_fixed_base_with_nearest_dp(self):
        import pandas as pd

        rows = pd.DataFrame(
            [
                {
                    "Architecture": "Vector",
                    "Memory": "GDDR6",
                    "Model": "Llama2-70B",
                    "Context": "128K",
                    "PP": 8,
                    "TP": 4,
                    "DP": 6,
                    "Total devices": 192,
                    "System throughput (tokens/s)": 1200.0,
                    "System power (W)": 3900.0,
                    "DGX H100 device-side power (W)": 4450.4,
                    "Absolute power delta vs DGX (W)": 550.4,
                    "Candidate closest integer DP": False,
                },
                {
                    "Architecture": "Vector",
                    "Memory": "GDDR6",
                    "Model": "Llama2-70B",
                    "Context": "128K",
                    "PP": 8,
                    "TP": 4,
                    "DP": 7,
                    "Total devices": 224,
                    "System throughput (tokens/s)": 1400.0,
                    "System power (W)": 4550.0,
                    "DGX H100 device-side power (W)": 4450.4,
                    "Absolute power delta vs DGX (W)": 99.6,
                    "Candidate closest integer DP": True,
                },
            ]
        )
        selected = combined_analysis.select_equal_power(rows)
        self.assertEqual(len(selected), 1)
        self.assertEqual(int(selected.iloc[0]["PP"]), 8)
        self.assertEqual(int(selected.iloc[0]["TP"]), 4)
        self.assertEqual(int(selected.iloc[0]["DP"]), 7)

    def test_base_objective_tie_prefers_smallest_replica(self):
        import pandas as pd

        rows = pd.DataFrame(
            [
                {
                    "Architecture": "Systolic 4x16",
                    "Memory": "GDDR6",
                    "Model": "Llama2-70B",
                    "Context": "32K",
                    "Can host one request": True,
                    "Throughput / device (tokens/s/device)": 10.0,
                    "Replica devices": 80,
                    "PP": 80,
                    "TP": 1,
                },
                {
                    "Architecture": "Systolic 4x16",
                    "Memory": "GDDR6",
                    "Model": "Llama2-70B",
                    "Context": "32K",
                    "Can host one request": True,
                    "Throughput / device (tokens/s/device)": 10.0,
                    "Replica devices": 40,
                    "PP": 40,
                    "TP": 1,
                },
            ]
        )
        selected = combined_analysis.select_base_objective(
            rows, "Throughput / device (tokens/s/device)"
        )
        self.assertEqual(int(selected.iloc[0]["PP"]), 40)


if __name__ == "__main__":
    unittest.main()
