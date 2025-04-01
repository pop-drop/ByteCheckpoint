import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from bytecheckpoint.utilities.ckpt_format.ckpt_loader import CKPTLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # The checkpoint path contains .metadata
    # For example: /mnt/disk/bytecheckpoint/fsdp_fast_ckpt/model
    parser.add_argument("--ckpt_path", required=True, type=str)
    args = parser.parse_args()

    url = args.ckpt_path
    loader = CKPTLoader(url)

    metadata = loader.load_metadata()

    for key, tensor in metadata.state_dict_metadata.items():
        print(f"key={key} value={tensor}")
    print(metadata.user_defined_dict)
