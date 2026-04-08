import os
import lm_eval

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def calc_ave_acc(out_result):
    # Don't change the order of this list.
    metric_list = ("acc_norm,none", "acc,none", "pass@1,create_test", "pass_at_1,none", "exact_match,strict-match", )
    for ben_name, result in out_result.items():
        for metric_name in metric_list:
            if metric_name in result:
                out_result[ben_name] = round(result[metric_name], 6)
                break
    out_result["summary_average"] = round(sum(out_result.values()) / max(len(out_result), 1), 6)
    return out_result


def eval(model_path, args):
    model_args = {"pretrained": model_path, "add_bos_token": args.add_bos_token, "dtype": args.dtype}
    results = lm_eval.simple_evaluate( 
        model=args.model_format,
        model_args=model_args,
        tasks=args.tasks,
        device=args.device,
        batch_size=args.batch_size,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        confirm_run_unsafe_code=True,
        #num_fewshot=0,
        #limit=5,
        )
    out_result = calc_ave_acc(results["results"])
    return out_result
