def convert_dict_to_query(config_dict, prefix="config"):
    query = {}
    for key, value in config_dict.items():
        if isinstance(value, dict):
            query.update(convert_dict_to_query(value, f"{prefix}.{key}"))
        else:
            query[f"{prefix}.{key}"] = value
    return query
