# fa23_yield

## Files:
**sharegpt_format_convert.ipynb:** pipeline for converting synthetic data format into ShareGPT conversation data format for fine-tuning.

**dataloader_mandrill_config.py:** This is the configuration file when you run a model (taken from HuggingFace/github) on Mandrill. If your data-set does not follow the specified format written in dataclass you can create a new custom dataclass and change the parameters based on your dataset. You would also need to add the custom dataclass name in NAME2CLS.

**synthetic.ipynb:** Generate pkl file dataset of AI acting as a Synthetic Environment
