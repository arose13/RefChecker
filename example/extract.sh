#!/bin/bash
refchecker-cli extract \
  --input_path example/example_in.json \
  --output_path example/example_out_claims.json \
  --extractor_name bedrock/anthropic.claude-3-sonnet-20240229-v1:0 \
  --extractor_max_new_tokens 1000
