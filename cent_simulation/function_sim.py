import torch
from utils import get_args, compare
from Llama import TransformerBlockLlama
from GPT import TransformerBlockGPT
from systolic_pim_microbench import microbench
from tp_mapping import kv_head_tp_shape, validate_kv_head_tp_role

if __name__ == "__main__":

    args = get_args()
    if args.filename:
        dic_model = torch.load(args.filename)
    else:
        head_dim = 128
        dim = head_dim * args.n_heads
        ffn_dim = args.ffn_dim
        tensor_device = "meta" if args.only_trace else None

        def model_zeros(shape):
            return torch.zeros(shape, device=tensor_device)

        TP_param = 8 if args.GPT3_175B_TP_8 else 1
        global_n_kv_heads = args.n_kv_heads if args.Llama_GQA else args.n_heads
        if args.kv_head_tp:
            if not args.model_parallel:
                raise ValueError("--kv-head-tp requires --model-parallel")
            shape = kv_head_tp_shape(
                dim=dim,
                query_heads=args.n_heads,
                kv_heads=global_n_kv_heads,
                ffn_dim=ffn_dim,
                tp=args.FC_devices,
                head_dim=head_dim,
            )
            validate_kv_head_tp_role(args.FC_devices, args.tp_device_role)
            TP_param = 1
            n_heads = shape.local_query_heads
            n_kv_heads = shape.local_kv_heads
            q_dim = shape.local_query_dim
            kv_dim = shape.local_kv_dim
            local_ffn_dim = shape.local_ffn_dim
        else:
            n_heads = args.n_heads // TP_param
            n_kv_heads = global_n_kv_heads if args.Llama_GQA else n_heads
            q_dim = dim // TP_param
            kv_dim = head_dim * n_kv_heads
            local_ffn_dim = ffn_dim // TP_param
        dic_model = {
            "TP_param": torch.tensor(TP_param),
            "dim": torch.tensor(dim),
            "n_heads": torch.tensor(n_heads),
            "global_n_heads": torch.tensor(args.n_heads),
            "global_n_kv_heads": torch.tensor(global_n_kv_heads),
            "head_dim": torch.tensor(head_dim),
            "x": model_zeros((args.batch_size, 1, dim)),
            "SANorm": model_zeros(dim),
            "FFNNorm": model_zeros(dim),
            "sa": model_zeros((args.batch_size, 1, dim)),
            "h": model_zeros((args.batch_size, 1, dim)),
            "out": model_zeros((args.batch_size, 1, dim)),
            "wq": model_zeros((q_dim, dim)),
            "wk": model_zeros((kv_dim, dim)),
            "wv": model_zeros((kv_dim, dim)),
            "xq": model_zeros((args.batch_size, 1, q_dim)),
            "xk": model_zeros((args.batch_size, 1, kv_dim)),
            "xv": model_zeros((args.batch_size, 1, kv_dim)),
            "start_pos": torch.tensor(args.seqlen - 1),
            "cache_k": model_zeros((args.batch_size, args.seqlen, n_kv_heads, head_dim)),
            "cache_v": model_zeros((args.batch_size, args.seqlen, n_kv_heads, head_dim)),
            "scores": model_zeros((args.batch_size, n_heads, 1, args.seqlen)),
            "output": model_zeros((args.batch_size, 1, q_dim)),
            "wo": model_zeros((dim, q_dim)) if args.kv_head_tp else model_zeros((dim // TP_param, dim)),
            "w1": model_zeros((local_ffn_dim, dim)),
            "w3": model_zeros((local_ffn_dim, dim)),
            "w2": model_zeros((dim, local_ffn_dim)) if args.kv_head_tp else model_zeros((dim // TP_param, ffn_dim)),
            "ffn": model_zeros((args.batch_size, 1, dim))
        }
    if args.Llama_GQA and "n_kv_heads" not in dic_model:
        dic_model["n_kv_heads"] = torch.tensor(
            args.n_kv_heads if args.filename else n_kv_heads
        )
    
    if args.systolic_pim_microbench:
        TB = microbench(dic_model, args)
    else:
        TB = TransformerBlockLlama(dic_model, args) if args.Llama_GQA or args.Llama or args.filename else TransformerBlockGPT(dic_model, args)
    # print("Variable\t Dimension\t\t\t Rows required\n")
    TB.memory_mapping()

    if args.only_trace:
        if args.embedding:
            TB.trace_only_embedding()
        elif args.only_FC:
            TB.trace_only_FC()
        elif args.systolic_pim_microbench:
            TB.generate_trace()
        elif args.systolic_pim:
            TB.trace_only_systolic_PIM()
        else:
            TB.trace_only()
        TB.finish()
        TB.file.close()
    elif args.pim_memory_mapping:
        dic_model = torch.load(args.filename)
        print("\n============ {} Functional Verification ============".format(args.filename.split("/")[-1].split(".")[0]))
        if args.systolic_pim:
            TB.memory_mapping_verification_systolic_pim()
            sa_aim = TB.self_attention_aim_systolic_pim()
            out_aim = TB.FFN_aim_systolic_pim(sa_aim)
        else:
            TB.memory_mapping_verification()
            sa_aim = TB.self_attention_aim()
            out_aim = TB.FFN_aim(sa_aim)
        compare(out_aim, TB.out, "AiM out")
        TB.finish()
        TB.file.close()
    else:
        dic_model = torch.load(args.filename)
        sa = TB.self_attention()
        out = TB.FFN(sa)
        compare(out, TB.out, "out")
        TB.file.close()
