import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from bytecheckpoint.utilities.ckpt_format.ckpt_loader import distcp_load_tool

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # The checkpoint path contains .metadata
    # For example: /mnt/disk/bytecheckpoint/fsdp_fast_ckpt/model
    parser.add_argument("--ckpt_path", required=True, type=str)
    # The file to read,
    # For example: __0_0.distcp
    parser.add_argument("--file_path", required=True, type=str)
    args = parser.parse_args()

    state_dict = distcp_load_tool(args.ckpt_path, args.file_path)

    for key, offset_tensor_list in state_dict.items():
        print(f"key={key} ")
        for kv_pair in offset_tensor_list:
            print(f"offset={kv_pair['offset']}, tensor_or_object={kv_pair['tensor_or_object']}")

# python3 scripts/load_ckpt.py --ckpt_path tmp_checkpoint_dir_fsdp/global_step_0/optimizer --file_path __0_0.distcp
