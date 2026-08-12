import math
from TransformerBlock import TransformerBlock

debug = True

class microbench(TransformerBlock):
    """
    TransformerBlock Class inherits computate functionality from PIM class
    """
    def __init__(self, dic_model, args):
        super().__init__(dic_model, args)

    def generate_trace(self):
        """
        Generate trace for microbenchmark of 8 * 8K * 8K matrix multiplication
        """

        row_index = 0
        seqlen = self.seqlen
        channel_list = [c for c in range(self.num_channels)]

        repeat = 1

        for _ in range(repeat):

            if "Vector" in self.microbench:
                # MAC Reduction Tree
                if "score" in self.microbench:
                    self.Vector_Matrix_Mul_score_pim_only_trace(row_index, seqlen, "breakdown_sa_score")
                elif "output" in self.microbench:
                    self.Vector_Matrix_Mul_output_pim_only_trace(row_index, seqlen, "breakdown_sa_output")
                else:
                    for i in range(8):
                        self.Vector_Matrix_Mul_weight_pim_only_trace(channel_list, row_index, 8192, 8192, self.FC_total_banks, "breakdown_sa_weight")
            elif "SA" in self.microbench:   # "SA8x16"  "SA4x16"  "SA2x16"  "SA1x16"
                # 8x16 Systolic Array
                if "score" in self.microbench:
                    self.Vector_Matrix_Mul_score_systolic_pim_only_trace(row_index, seqlen)
                elif "output" in self.microbench:
                    self.Vector_Matrix_Mul_output_systolic_pim_only_trace(row_index, seqlen)
                else:
                    iterations = max(1, 8 // self.systolic_dim)
                    for i in range(iterations):
                        self.Vector_Matrix_Mul_weight_systolic_pim_only_trace(channel_list, row_index, 8192, 8192)
