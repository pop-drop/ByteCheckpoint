import argparse

from torch.distributed.checkpoint.metadata import MetadataIndex

from bytecheckpoint.utilities.ckpt_format.ckpt_loader import CKPTLoader


def user_defined_modify_rule_function(key: str) -> str:
    """
    Add you own logic here, the tool will pass all keys one by one to this function.
    For each key, the tool will call this function.
    If you want to modify the key, return the new key.
    If you don't want to modify, do nothing. Let the last line return the origin key.

    Example Code:
    if "language_model." in key:
        return key.replace("language_model.", "gpt.")
    """

    return key


def modify_keys(metadata):
    state_dict_keys_to_del = []
    state_dict_new_kv = {}
    for key, value in metadata.state_dict_metadata.items():
        new_key = user_defined_modify_rule_function(key)
        if new_key != key:
            state_dict_new_kv[new_key] = value
            state_dict_keys_to_del.append(key)
            print("current key: ", key)
            print("new key:", new_key)
    for state_dict_key in state_dict_keys_to_del:
        del metadata.state_dict_metadata[state_dict_key]
    metadata.state_dict_metadata.update(state_dict_new_kv)

    # Storage Metadata
    storage_keys_to_del = []
    storage_new_kv = {}
    for key, value in metadata.storage_data.items():
        new_fqn = user_defined_modify_rule_function(key.fqn)
        if new_fqn != key.fqn:
            new_key = MetadataIndex(fqn=new_fqn, offset=key.offset, index=key.index)
            storage_new_kv[new_key] = value
            storage_keys_to_del.append(key)
            print("current key fqn: ", key.fqn)
            print("new key fqn:", new_key.fqn)
    for storage_key in storage_keys_to_del:
        del metadata.storage_data[storage_key]
    metadata.storage_data.update(storage_new_kv)

    # Tensor metadata
    tensor_keys_to_del = []
    tensor_new_kv = {}
    for key, value in metadata.all_tensor_metadata.items():
        new_fqn = user_defined_modify_rule_function(key.fqn)
        if new_fqn != key.fqn:
            new_key = MetadataIndex(fqn=new_fqn, offset=key.offset, index=key.index)
            tensor_new_kv[new_key] = value
            tensor_keys_to_del.append(key)
            print("current key's fqn: ", key.fqn)
            print("new key's fqn:", new_key.fqn)
    for tensor_key in tensor_keys_to_del:
        del metadata.all_tensor_metadata[tensor_key]
    metadata.all_tensor_metadata.update(tensor_new_kv)

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True, type=str)
    # The output file name
    parser.add_argument("--result_name", required=True, type=str)

    args = parser.parse_args()
    url = args.ckpt_path
    loader = CKPTLoader(url)

    metadata = loader.load_metadata()

    metadata = modify_keys(metadata)

    import pickle

    with open(args.result_name, "wb") as file:
        pickle.dump(metadata, file)
